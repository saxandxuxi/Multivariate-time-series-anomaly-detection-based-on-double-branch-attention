import math

import torch
from torch import nn
import torch.nn.functional as F


class FeatureCorrelation(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim  # 输入特征维度（M或d_model）

        # 用于生成关联矩阵的变换层
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x):
        """
        x: [B, L, feature_dim] - 输入特征（原始x或嵌入后的x_ori）
        返回: 特征关联矩阵 [B, L, L]
        """
        B, L, _ = x.shape

        # 特征变换（增强特征表达能力）
        transformed_x = self.transform(x)  # [B, L, feature_dim]

        # 计算特征间关联矩阵（使用点积相似度）
        corr_matrix = torch.bmm(transformed_x, transformed_x.transpose(1, 2)) / math.sqrt(self.feature_dim)
        corr_matrix = F.softmax(corr_matrix, dim=-1)  # 归一化到概率分布 [B, L, L]

        return corr_matrix
#
class AnomalyAwareFusion(nn.Module):
    """修正：返回与输入相同形状的更新特征"""

    def __init__(self, dim, patch_size, patch_num, num_heads=8, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = patch_num

        # 特征间注意力（更新patch_size维度）
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 补丁间注意力（更新patch_num维度）
        self.patch_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 异常感知权重计算
        self.anomaly_weight = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_patch_size, x_patch_num, x_ori):
        # x_patch_size shape: [B*M, patch_num, D]
        # x_patch_num shape: [B*M, patch_size, D]
        # x_ori shape: [B, win_size, D]

        B_times_M, _, D = x_patch_size.shape
        B, win_size, D = x_ori.shape
        M = B_times_M // B

        # 1. 特征间注意力：更新x_patch_size
        x_patch_size_seq = x_patch_size.transpose(0, 1)  # [patch_num, B*M, D]
        x_patch_size_attn, feature_attn_weights = self.feature_attn(
            x_patch_size_seq,  # query
            x_ori.reshape(B, win_size, D).expand(M, -1, -1, -1).reshape(B * M, win_size, D).transpose(0, 1),  # key
            x_ori.reshape(B, win_size, D).expand(M, -1, -1, -1).reshape(B * M, win_size, D).transpose(0, 1)  # value
        )  # [patch_num, B*M, D]

        x_patch_size_updated = x_patch_size_attn.transpose(0, 1)  # [B*M, patch_num, D]

        # 2. 补丁间注意力：更新x_patch_num
        x_patch_num_seq = x_patch_num.transpose(0, 1)  # [patch_size, B*M, D]
        x_patch_num_attn, patch_attn_weights = self.patch_attn(
            x_patch_num_seq,  # query
            x_ori.reshape(B, win_size, D).expand(M, -1, -1, -1).reshape(B * M, win_size, D).transpose(0, 1),  # key
            x_ori.reshape(B, win_size, D).expand(M, -1, -1, -1).reshape(B * M, win_size, D).transpose(0, 1)  # value
        )  # [patch_size, B*M, D]

        x_patch_num_updated = x_patch_num_attn.transpose(0, 1)  # [B*M, patch_size, D]

        # 3. 异常感知权重计算
        patch_repr = x_patch_size_updated.mean(dim=1)  # [B*M, D]
        anomaly_weights = self.anomaly_weight(patch_repr)  # [B*M, 1]

        # 4. 应用异常权重
        x_patch_size_weighted = x_patch_size_updated * anomaly_weights.unsqueeze(1)
        x_patch_num_weighted = x_patch_num_updated * anomaly_weights.unsqueeze(1)

        # 返回更新后的特征和注意力权重
        return {
            'x_patch_size': x_patch_size_weighted,  # [B*M, patch_num, D]
            'x_patch_num': x_patch_num_weighted,  # [B*M, patch_size, D]
            'attention_weights': {
                'anomaly_weights': anomaly_weights.reshape(B, M, 1),  # [B, M, 1]
                'feature_attn_weights': feature_attn_weights.reshape(B, M, self.patch_num, win_size),
                # [B, M, patch_num, win_size]
                'patch_attn_weights': patch_attn_weights.reshape(B, M, self.patch_size, win_size)
                # [B, M, patch_size, win_size]
            }
        }


 # class FeatureCorrelation(nn.Module):
#     def __init__(self, enc_in, patch_sizes, hidden_dim, dropout=0.1):
#         super().__init__()
#         self.enc_in = enc_in
#         self.patch_sizes = patch_sizes
#         self.transforms = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(enc_in, hidden_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, enc_in)
#             )
#             for _ in patch_sizes
#         ])
#
#     def forward(self, x):
#         """
#         x: [B, L, M] - 输入特征
#         返回: 不同补丁大小下的关联矩阵列表
#         """
#         B, L, M = x.shape
#         correlation_matrices = []
#
#         for i, patch_size in enumerate(self.patch_sizes):
#             # 划分子序列
#             splits = L // patch_size
#             if splits == 0:
#                 splits = 1
#
#             # 计算每个子序列的特征表示
#             x_patches = x.reshape(B, splits, patch_size, M)  # [B, splits, patch_size, M]
#             patch_features = x_patches.mean(dim=2)  # [B, splits, M]
#
#             # 特征变换
#             transformed_features = self.transforms[i](patch_features)  # [B, splits, M]
#
#             # 计算特征间关联矩阵
#             corr_matrix = torch.bmm(transformed_features.transpose(1, 2), transformed_features) / math.sqrt(
#                 M)  # [64,22,22]
#             corr_matrix = F.softmax(corr_matrix, dim=-1)  # 归一化
#
#             correlation_matrices.append(corr_matrix)
#
#         return correlation_matrices