import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .Mahalanobis_mask import Mahalanobis_mask, PyramidMahalanobisMask, MultiDistanceChannelClustering
from .TCM import Linear_extractor_cluster
from .attn import DAC_structure, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN
from tkinter import _flatten

# def order_shuffle(x, size=None):
#     """对通道维度进行随机混洗（通道维度在最后一维）"""
#     if size is None:
#         size = x.shape[-1]  # 通道维度在最后一维（B, L, M）
#     shuffled_indices = torch.randperm(size, device=x.device)
#     return x[..., shuffled_indices]  # 混洗最后一维（通道维度）

def order_shuffle(tensor, size=2, dim=1):
    """将指定维度的特征向量均分为两部分并交换顺序"""
    target_size = tensor.shape[dim]
    half_n = target_size // size
    part1 = tensor.narrow(dim, half_n, target_size - half_n)  # 后半部分
    part2 = tensor.narrow(dim, 0, half_n)                     # 前半部分
    return torch.cat([part1, part2], dim=dim)  # 拼接：后半部分+前半部分


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index,channel_mask):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=channel_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True,num_experts = 3,k=2):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size
        self.channel = channel
        self.win_size = win_size
        self.num_experts = num_experts
        self.k = k

        # Patching List
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size // patchsize, d_model, dropout))

        self.embedding_window_size = DataEmbedding(1, d_model, dropout)#原来是enc_in
        # 新增：为每个尺度的补丁间(series)和补丁内(prior)分别定义中心向量
        self.series_centers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, win_size)) for _ in range(len(patch_size))
        ])
        self.prior_centers = nn.ParameterList([
            nn.Parameter(torch.zeros(1, win_size)) for _ in range(len(patch_size))
        ])

        # 初始化中心向量（可选）
        for i in range(len(patch_size)):
            nn.init.kaiming_uniform_(self.series_centers[i])
            nn.init.kaiming_uniform_(self.prior_centers[i])

        # 新增：Channel Clustering Module (CCM)
        self.cross_channel_ffn =MultiDistanceChannelClustering(win_size)
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(win_size, patch_size, channel, False, attention_dropout=dropout,
                                  output_attention=output_attention),
                    d_model, patch_size, channel, n_heads, win_size) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        B, L, M = x.shape  # Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        revin_layer = RevIN(num_features=M)
        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        # x = x.permute(0, 2, 1).reshape(B * M, L, 1)
        # x, CVloss = self.tcm(x)
        # x = x.view(B, M, -1).permute(0, 2, 1)

        x_channel = x.permute(0, 2, 1)
        channel_mask = self.cross_channel_ffn(x_channel)#(bs,1,22,22)
        x_embed = x_channel.reshape(B*M, L, 1)
        x_ori = self.embedding_window_size(x_embed)#(1408,70,256)
        # # 1. 新增：跨通道注意力处理（移至编码器前）
        # x_for_cross = x_ori.transpose(0, 1)  # [L, B, d_model]
        # cross_output, attn_weights = self.cross_channel_attn(
        #     x_for_cross, x_for_cross, x_for_cross
        # )  # [L, B, d_model], [B, L, L]
        # x_ori = cross_output.transpose(0, 1)  # 转回 [B, L, d_model]

        # 补丁间的x_patch_size:(B,patch_num,256) 补丁内的x_patch_num:(B,patch_size,256)
        # Mutil-scale Patching Operation
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x,x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l')  # Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l')  # Batch channel win_size
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p=patchsize)#Batch*channel n p
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)#Batch*channel n d_model
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p=patchsize)#Batch*channel p n
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)#Batch*channel p d_model
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index,channel_mask)  # [B,1,L,L]
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))  ##[B,1,L,L]
        prior_patch_mean = list(_flatten(prior_patch_mean))
        # 新增：从扁平化列表中提取特征（假设每个尺度对应一个中心向量）
        series_features = []
        prior_features = []

        # 假设每个尺度的层数相同，按尺度分组处理
        layers_per_scale = len(series_patch_mean) // len(self.patch_size)
        for scale_idx in range(len(self.patch_size)):
            start_idx = scale_idx * layers_per_scale
            end_idx = (scale_idx + 1) * layers_per_scale

            # 提取当前尺度的所有层的特征
            scale_series_tensors = series_patch_mean[start_idx:end_idx]
            scale_prior_tensors = prior_patch_mean[start_idx:end_idx]

            # 聚合当前尺度的所有层的特征为 [B, L]
            scale_series_feature = torch.stack([
                tensor.squeeze(1).mean(dim=2) for tensor in scale_series_tensors
            ], dim=0).mean(dim=0)  # [B, L]

            scale_prior_feature = torch.stack([
                tensor.squeeze(1).mean(dim=2) for tensor in scale_prior_tensors
            ], dim=0).mean(dim=0)  # [B, L]

            series_features.append(scale_series_feature)
            prior_features.append(scale_prior_feature)

        if self.output_attention:
            return series_patch_mean, prior_patch_mean, series_features, prior_features, self.series_centers, self.prior_centers
        else:
            return None
        

