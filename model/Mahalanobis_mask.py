import numpy as np
import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import torch.fft
from einops import rearrange
from torch.distributions.gumbel import Gumbel

from model.compute_ppr import compute_ppr


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)#初始化可学习的马氏距离矩阵A（257,257），512的一半

    #  通道聚类模块（CCM）
    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))#实值FFT，取模得到频域幅值
        X1 = XF.unsqueeze(2)#（32,7,1,257）
        X2 = XF.unsqueeze(1)#（32,1,7,257），形状 [B, C, C, D]，通道i与j的频域差异
        # B x C x C x D
        diff = X1 - X2#（32,7,7,257）
        # 可学习马氏距离计算
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)#（32,7,7,257）

        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)#（32,7,7）

        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10)## 距离倒数，转换为相似度，（32,7,7）
        # 对角线置零

        identity_matrices = 1 - torch.eye(exp_dist.shape[-1])#（7,7）
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device)#（32,7,7）
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)#（32,7,7），# 移除对角线元素（i=j时设为0）
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)# # 归一化分母
        exp_max = exp_max.detach()## 概率矩阵归一化

        # B x C x C
        p = exp_dist / exp_max#（32,7,7）

        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)

        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99# 恢复P_ii=1，并缩放至[0, 0.99]区间

        return p
    # 通过Gumbel-Softmax 重参数化从概率矩阵p中采样离散的二值掩码M
    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape# distribution_matrix即概率矩阵P，形状[B, C, C],(32,7,7)

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')# # 展平为[B*C*C, 1][1568,1]
        r_flatten_matrix = 1 - flatten_matrix# # 非相关概率 1-P_ij

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)# # log(P_ij / (1-P_ij))
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)# # log((1-P_ij)/P_ij)
        # Gumbel 噪声注入与重参数化
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)#形状[B*C*C, 2],[1568,2]
        resample_matrix = gumbel_softmax(new_matrix, hard=True)#[1568,2]

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)#[32,7,7]
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)#CCM的概率矩阵

        # bernoulli中两个通道有关系的概率
        sample = self.bernoulli_gumbel_rsample(p)#CCM的最终输出（32,7,7）
        sample = compute_ppr(sample)
        mask = sample.unsqueeze(1)#（32，1,7,7）

        return mask


class PyramidMahalanobisMask(nn.Module):
    def __init__(self, input_size, num_bands=3, channel_count=22):
        super().__init__()
        self.frequency_size = input_size // 2 + 1
        self.num_bands = num_bands
        self.channel_count = channel_count

        # 1. 计算每个子带的大小
        band_size = self.frequency_size // num_bands
        self.band_sizes = [band_size] * (num_bands - 1)
        self.band_sizes.append(self.frequency_size - band_size * (num_bands - 1))  # 最后一个子带处理余数

        # 2. 为每个子带创建独立的马氏距离矩阵
        self.A_bands = nn.ParameterList([
            nn.Parameter(torch.randn(s, s)) for s in self.band_sizes
        ])

        # 3. 子带融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(num_bands))

        # 4. 自适应阈值参数（基于通道数）
        self.register_buffer('threshold',
                             torch.tensor(0.7 - 0.02 * (channel_count - 8)))

    def calculate_prob_distance(self, X):
        device = X.device

        # 1. 频域变换
        XF = torch.abs(torch.fft.rfft(X, dim=-1))  # [B, C, F](62,22,36)

        # 2. 划分子带
        XF_bands = torch.split(XF, self.band_sizes, dim=-1)

        # 3. 为每个子带计算相似度矩阵
        dist_bands = []
        for i, (band, A) in enumerate(zip(XF_bands, self.A_bands)):
            # 计算通道间差异
            X1 = band.unsqueeze(2)  # [B, C, 1, band_size]
            X2 = band.unsqueeze(1)  # [B, 1, C, band_size]
            diff = X1 - X2  # [B, C, C, band_size]

            # 子带特定的马氏距离计算
            temp = torch.einsum("dk,bxck->bxcd", A.to(device), diff)  # [B, C, C, band_size]
            dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)  # [B, C, C]

            dist_bands.append(dist)

        # 4. 自适应权重融合
        weights = F.softmax(self.fusion_weights, dim=0).unsqueeze(0).unsqueeze(0)  # [1, 1, num_bands]
        dist_weighted = torch.stack(dist_bands, dim=-1) * weights.to(device)  # [B, C, C, num_bands]
        dist = torch.sum(dist_weighted, dim=-1)  # [B, C, C]

        # 5. 转换为相似度矩阵
        exp_dist = 1 / (dist + 1e-10)

        # 6. 对角线置零
        identity = 1 - torch.eye(exp_dist.shape[-1], device=device)
        mask = identity.repeat(exp_dist.shape[0], 1, 1)
        exp_dist = exp_dist * mask

        # 7. 基于自适应阈值的归一化
        exp_max = exp_dist.max(dim=-1, keepdim=True)[0].detach()
        p = exp_dist / exp_max

        # 8. 恢复自连接并缩放
        diag = torch.eye(p.shape[-1], device=device)
        p = (p * mask + diag) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        # 与原代码相同的Gumbel-Softmax采样
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = F.gumbel_softmax(new_matrix, hard=True, tau=1.0)

        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)
        sample = self.bernoulli_gumbel_rsample(p)
        return sample.unsqueeze(1)  # [B, 1, C, C]


class MultiDistanceChannelClustering(nn.Module):
    """支持多种距离度量的通道聚类模块，可替代原有的马氏距离实现"""

    def __init__(self, input_size, distance_type='rbf', alpha=1.5, sigma=1.5,
                 dtw_window=3, num_bands=1, channel_count=None):
        """
        参数:
            input_size: 输入序列长度
            distance_type: 距离类型，可选'mahalanobis', 'complex', 'rbf', 'dtw', 'wasserstein'
            alpha: 复数域距离中振幅的权重(0-1)
            sigma: RBF核的带宽参数
            dtw_window: DTW动态规划的窗口大小
            num_bands: 频域子带数量
            channel_count: 通道数量，用于自适应阈值计算
        """
        super().__init__()
        self.frequency_size = input_size // 2 + 1
        self.distance_type = distance_type
        self.alpha = alpha
        self.sigma = sigma
        self.dtw_window = dtw_window
        self.num_bands = num_bands

        # 根据距离类型初始化不同的参数
        if distance_type == 'mahalanobis':
            self.A = nn.Parameter(torch.randn(self.frequency_size, self.frequency_size))
        elif distance_type == 'rbf':
            self.gamma = nn.Parameter(torch.ones(1))  # 可学习的RBF核缩放因子
        elif distance_type == 'complex':
            self.alpha = nn.Parameter(torch.tensor(alpha))  # 可学习的振幅权重

        # 子带划分
        if num_bands > 1:
            band_size = self.frequency_size // num_bands
            self.band_sizes = [band_size] * (num_bands - 1)
            self.band_sizes.append(self.frequency_size - band_size * (num_bands - 1))
            self.band_weights = nn.Parameter(torch.ones(num_bands))

        # 自适应阈值
        if channel_count is not None:
            self.register_buffer('threshold', torch.tensor(0.7 - 0.02 * (channel_count - 8)))
        else:
            self.register_buffer('threshold', torch.tensor(0.7))

    def complex_distance(self, X1,X2):
        """复数域距离：结合振幅和相位信息"""
        # 振幅和相位差异计算（如上述流程）
        amp1 = torch.abs(X1)
        phase1 = torch.angle(X1)
        amp2 = torch.abs(X2)
        phase2 = torch.angle(X2)

        amp_diff = torch.abs(amp1 - amp2)
        phase_diff = torch.min(torch.abs(phase1 - phase2), 2 * np.pi - torch.abs(phase1 - phase2))

        combined_diff = self.alpha * amp_diff + (1 - self.alpha) * phase_diff
        dist = torch.sum(combined_diff, dim=-1)  # 沿频率维度求和

        return dist

    def rbf_distance(self, X1, X2):
        """RBF核距离"""
        diff = X1 - X2
        euclidean_dist = torch.sum(diff ** 2, dim=-1)
        return 1 - torch.exp(-euclidean_dist / (2 * self.sigma ** 2))

    def dtw_distance(self, X1, X2):
        """频域DTW距离（简化实现）"""
        n, m = X1.shape[-1], X2.shape[-1]
        dtw_matrix = torch.zeros(X1.shape[0], X1.shape[1], X1.shape[2], n, m).to(X1.device)

        # 初始化第一行和第一列
        for i in range(n):
            for j in range(m):
                if abs(i - j) > self.dtw_window:  # 窗口约束
                    dtw_matrix[..., i, j] = float('inf')
                    continue

                cost = torch.abs(X1[..., i] - X2[..., j])
                if i == 0 and j == 0:
                    dtw_matrix[..., i, j] = cost
                elif i == 0:
                    dtw_matrix[..., i, j] = cost + dtw_matrix[..., i, j - 1]
                elif j == 0:
                    dtw_matrix[..., i, j] = cost + dtw_matrix[..., i - 1, j]
                else:
                    dtw_matrix[..., i, j] = cost + torch.min(
                        torch.stack([
                            dtw_matrix[..., i - 1, j],
                            dtw_matrix[..., i, j - 1],
                            dtw_matrix[..., i - 1, j - 1]
                        ], dim=-1),
                        dim=-1
                    )[0]

        return dtw_matrix[..., -1, -1]

    def calculate_distance(self, XF):
        """根据所选距离类型计算通道间距离矩阵"""
        X1 = XF.unsqueeze(2)  # [B, C, 1, F](64,22,1,36)
        X2 = XF.unsqueeze(1)  # [B, 1, C, F](64,1,22,36)

        if self.distance_type == 'mahalanobis':
            # 马氏距离
            diff = X1 - X2
            temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
            dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)
        elif self.distance_type == 'complex':
            dist = self.complex_distance(X1,X2)
        elif self.distance_type == 'rbf':
            # RBF核距离
            dist = self.rbf_distance(X1, X2)
        elif self.distance_type == 'dtw':
            # 频域DTW距离
            dist = self.dtw_distance(X1, X2)
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

        return dist

    def calculate_distance_with_bands(self, X):
        """划分子带计算距离并融合"""
        # 频域变换
        XF = torch.abs(torch.fft.rfft(X, dim=-1))

        if self.num_bands == 1:
            return self.calculate_distance(XF)

        # 划分子带
        XF_bands = torch.split(XF, self.band_sizes, dim=-1)
        band_distances = []

        for band in XF_bands:
            band_dist = self.calculate_distance(band)
            band_distances.append(band_dist)

        # 带权重融合
        band_weights = F.softmax(self.band_weights, dim=0)
        weighted_distances = [w * d for w, d in zip(band_weights, band_distances)]
        return torch.sum(torch.stack(weighted_distances, dim=0), dim=0)

    def forward(self, X):
        """
        输入:
            X: 输入张量 [B, T, C]

        输出:
            mask: 通道聚类掩码 [B, 1, C, C]
        """
        # 计算距离矩阵
        dist = self.calculate_distance_with_bands(X)#(64,22,22)

        # 转换为相似度矩阵
        exp_dist = 1 / (dist + 1e-10)

        # 对角线置零
        batch_size = exp_dist.shape[0]
        identity = 1 - torch.eye(exp_dist.shape[-1], device=exp_dist.device)
        mask = identity.repeat(batch_size, 1, 1)
        exp_dist = exp_dist * mask

        # 归一化
        exp_max = exp_dist.max(dim=-1, keepdim=True)[0].detach()
        p = exp_dist / exp_max

        # 恢复自连接并缩放
        diag = torch.eye(p.shape[-1], device=p.device)
        p = (p * mask + diag) * 0.99

        # Gumbel-Softmax采样
        gumbel = Gumbel(0, 1).sample(p.shape).to(p.device)
        logits = torch.log(p + 1e-10) - torch.log(1 - p + 1e-10) + gumbel
        sample = torch.sigmoid(logits)

        # 二值化（hard=True）
        hard_sample = (sample > self.threshold).float()
        hard_sample = hard_sample - sample.detach() + sample  # 保留梯度路径

        return hard_sample.unsqueeze(1)  # [B, 1, C, C]



if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    x= torch.randn(64,22,70)
    model1 = Mahalanobis_mask(70)
    # model2 = PyramidMahalanobisMask(70).to('cuda:0')
    # model3 = MultiDistanceChannelClustering(70).to('cuda:0')
    y = model1(x)
    # y2 = model2(x)
    # y3 = model3(x)
    print(y.shape)#[64,1,22,22]
    print(y)