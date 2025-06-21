import numpy as np
import torch.nn as nn
import torch
from math import sqrt
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import torch.fft
from einops import rearrange
from torch.distributions.gumbel import Gumbel
def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))#随机选取一半的算子
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=64, H=8,mode_select_method='random'):
        super(FourierBlock, self).__init__()
        """
        1D Fourier block for input shape (bs, channel, seq_len, d_model)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.mode_select_method = mode_select_method

        self.scale = (1 / (in_channels * out_channels))
        # 调整权重矩阵形状：移除头维度，合并通道与模型维度
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(H, in_channels//H, out_channels//H, modes, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, x, channel):
        # size = [B, L, H, E]
        B, L, H, E = x.shape  # 3,96,8,2/3,144,8,2
        x = x.permute(0, 2, 3, 1)  # 3,8,2,96/3,8,2,144 (B,H,E,L)
        # 动态获取频域模式索引
        self.index = get_frequency_modes(L, modes=self.modes, mode_select_method=self.mode_select_method)
        # print(self.index)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)  # 3,8,2,49
        # Perform Fourier neural operations频域特征变换
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)  # (3,8,2,49)/3,8,2,73
        for wi, i in enumerate(self.index):
            # 提取当前频域模式
            xq_ft = x_ft[:, :, :, i]  # (bs, channel, d_model)
            w = self.weights1[:, :, :, wi]  # (1, channel, out_channels)
            # 复数矩阵乘法（带单头维度）
            out_ft[:, :, :, wi] = self.compl_mul1d(xq_ft, w)# Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))  # out_ft:(3,8,2,49)->x(3,8,2,96)/(3,8,2,73)->x(3,8,2,144)
        x = x.permute(0, 3, 1, 2)
        return x



if __name__ == '__main__':
    seq_len = 35
    in_channels = 256
    out_channels = 256
    x = torch.randn(64,22, 35, 256)
    model = FourierBlock(in_channels, out_channels)
    y = model(x,channel=22)
    print(y.shape)


# class FourierBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, seq_len, modes=64, mode_select_method='random'):
#         super(FourierBlock, self).__init__()
#         # print('fourier enhanced block used!')
#         """
#         1D Fourier block for input shape (bs, channel, seq_len, d_model)
#         """
#         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
#         print('modes={}, index={}'.format(modes, self.index))
#
#         self.scale = (1 / (in_channels * out_channels))
#         # 调整权重矩阵形状：移除头维度，合并通道与模型维度
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(1, in_channels, out_channels, len(self.index), dtype=torch.cfloat)
#         )
#     # Complex multiplication
#     def compl_mul1d(self, input, weights):
#         # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
#         return torch.einsum("bhi,hio->bho", input, weights)
#
#     def forward(self, x):
#         # 输入形状: (bs, channel, seq_len, d_model)
#         B, C, L, D = x.shape
#         # 调整维度为 (bs, channel, d_model, seq_len) 以便沿序列维度做FFT
#         x_reshaped = x.permute(0, 1, 3, 2)  # (bs, channel, d_model, seq_len)
#
#         # 计算傅里叶系数
#         x_ft = torch.fft.rfft(x_reshaped, dim=-1)  # (bs, channel, d_model, seq_len//2+1)
#         # 频域特征变换
#         out_ft = torch.zeros(B, C, D, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         for wi, i in enumerate(self.index):
#             # 提取当前频域模式
#             xq_ft = x_ft[:, :, :, i]  # (bs, channel, d_model)
#             w = self.weights1[:, :, :, wi]  # (1, channel, out_channels)
#             # 复数矩阵乘法（带单头维度）
#             out_ft[:, :, :, wi] = self.compl_mul1d(xq_ft, w)
#
#         # 逆傅里叶变换转回时域
#         x = torch.fft.irfft(out_ft, n=x_reshaped.size(-1))  # (bs, channel, d_model, seq_len)
#
#         # 调整回输入形状 (bs, channel, seq_len, d_model)
#         x = x.permute(0, 1, 3, 2)
#         return x
