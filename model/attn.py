import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from einops import rearrange, reduce, repeat
from model.DynamicReconHead import DynamicReconHead


#嵌入之后到这个模块进行变换
class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05,
                 output_attention=False):
        """
            win_size：窗口大小，代表输入序列的长度。
            patch_size：补丁大小，用于将输入序列划分为多个补丁。
            channel：通道数，可理解为特征的维度，就是有多少个传感器
            mask_flag：布尔值，是否使用掩码，默认为 True。
            scale：缩放因子，默认为 None。
            attention_dropout：注意力机制中的丢弃率，默认为 0.05。
            output_attention：布尔值，是否输出注意力分数，默认为 False。
            self.dropout：创建一个 nn.Dropout 层，用于在注意力计算过程中随机丢弃部分元素，防止过拟合。
        """
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index,
                patch_size_mask=None,  # 新增！形状: (B, 1, L_patch_num, L_patch_num)
                patch_num_mask=None):

        # Patch-wise Representation
        B, L, H, E = queries_patch_size.shape  # batch_size, patch_num, n_head, d_model/n_head
        scale_patch_size = self.scale or 1. / sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size,
                                         keys_patch_size)  # batch, nheads, p_num, p_num
        if self.mask_flag:
            # scores_patch_size = scores_patch_size.masked_fill(patch_size_mask == 0, -torch.inf)
            scores_patch_size = scores_patch_size * patch_size_mask
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))  # B H N N

        # In-patch Representation
        B, L, H, E = queries_patch_num.shape  # batch_size, patch_size, n_head, d_model/n_head
        scale_patch_num = self.scale or 1. / sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num,
                                        keys_patch_num)  # batch, nheads, p_size, p_size
        if self.mask_flag:
            # 掩码形状 (B,1,L,L) → 广播到 (B,H,L,L)
            # scores_patch_num = scores_patch_num.masked_fill(patch_num_mask == 0, -torch.inf)
            scores_patch_num = scores_patch_num * patch_num_mask
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1))  # B H S S

        # 上采样到window_size
        target_size = self.window_size
        series_patch_size = F.interpolate(series_patch_size, size=(target_size, target_size), mode='bilinear',
                                          align_corners=False)
        series_patch_num = F.interpolate(series_patch_num, size=(target_size, target_size), mode='bilinear',
                                         align_corners=False)

        # 通道间平均
        # series_patch_size = series_patch_size.mean(dim=1, keepdim=True)  # bs, 1, patch_num, patch_num
        # series_patch_num = series_patch_num.mean(dim=1, keepdim=True)  # bs, 1, patch_size, patch_size

        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return (None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        """
            attention：一个注意力机制的实例，用于后续的注意力计算。
            d_model：模型的嵌入维度，即输入特征的维度。
            patch_size：补丁的大小，用于将输入序列划分为多个补丁。
            channel：通道数，可理解为特征的维度数量。
            n_heads：多头注意力的头数。
            win_size：窗口大小，代表输入序列的长度。
            d_keys 和 d_values：分别是键（keys）和值（values）的维度，默认为 None。
        """
        # 首先调用父类nn.Module的初始化方法
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        # self.fourier_query_enhanced1 = FourierBlock(d_model,d_model,H=n_heads)#查询增强
        # self.fourier_key_enhanced2 = FourierBlock(d_model,d_model,H=n_heads)#键增强
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, patch_index,
                patch_size_mask=None,  # 新增！形状: (B, 1, L_patch_num, L_patch_num)
                patch_num_mask=None):
        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(B, L, H, -1)
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(B, L, H, -1)  # (B,L,H,D_model)
        # queries_patch_size = self.fourier_query_enhanced1(queries_patch_size,channel=self.channel)
        # keys_patch_size = self.fourier_key_enhanced2(keys_patch_size,channel=self.channel)
        # patch_num
        B, L, M = x_patch_num.shape
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(B, L, H, -1)
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(B, L, H, -1)
        # queries_patch_num = self.fourier_query_enhanced1(queries_patch_num,channel=self.channel)
        # keys_patch_num = self.fourier_key_enhanced2(keys_patch_num,channel=self.channel)

        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            None, patch_index,
            patch_size_mask=patch_size_mask, patch_num_mask=patch_num_mask
        )

        return series, prior

  # # 处理series_patch_size
  #       batch_times_heads, heads, n, n = series_patch_size.shape
  #       series_patch_size = series_patch_size.view(batch_times_heads, heads, n, n)
  #
  #       # 使用interpolate而不是expand，更稳健
  #       series_patch_size = F.interpolate(
  #           series_patch_size,
  #           size=(self.window_size, self.window_size),
  #           mode='nearest'
  #       )
  #
  #       # 对series_patch_num执行相同操作
  #       batch_times_heads, heads, s, s = series_patch_num.shape
  #       series_patch_num = series_patch_num.view(batch_times_heads, heads, s, s)
  #       series_patch_num = F.interpolate(
  #           series_patch_num,
  #           size=(self.window_size, self.window_size),
  #           mode='nearest'
  #       )
 # series_patch_size = series_patch_size.unsqueeze(2)
 #        series_patch_num = series_patch_num.unsqueeze(2)
 #
 #        # 重塑为 (batch, channels, height, width) 格式，以便进行插值
 #        batch_times_channels, heads, channels, n, n = series_patch_size.shape
 #        series_patch_size_reshaped = series_patch_size.view(batch_times_channels * heads, channels, n, n)
 #
 #        batch_times_channels, heads, channels, s, s = series_patch_num.shape
 #        series_patch_num_reshaped = series_patch_num.view(batch_times_channels * heads, channels, s, s)
 #
 #        # 使用最近邻插值进行上采样，确保输出尺寸为win_size x win_size
 #        series_patch_size_upsampled = nn.functional.interpolate(
 #            series_patch_size_reshaped,
 #            size=(self.window_size, self.window_size),
 #            mode='nearest'
 #        )
 #
 #        series_patch_num_upsampled = nn.functional.interpolate(
 #            series_patch_num_reshaped,
 #            size=(self.window_size, self.window_size),
 #            mode='nearest'
 #        )
 #
 #        # 重塑回原始的维度结构
 #        series_patch_size = series_patch_size_upsampled.view(
 #            batch_times_channels, heads,
 #            channels,
 #            self.window_size,
 #            self.window_size
 #        ).squeeze(2)  # 移除通道维度
 #
 #        series_patch_num = series_patch_num_upsampled.view(
 #            batch_times_channels, heads,
 #            channels,
 #            self.window_size,
 #            self.window_size
 #        ).squeeze(2)  # 移除通道维度
# # 记录原始维度
#         batch_times_channels, heads, n, n = series_patch_size.shape
#         _, _, s, s = series_patch_num.shape
#
#         # 重塑张量，将batch和heads合并
#         series_patch_size_reshaped = series_patch_size.view(batch_times_channels * heads, 1, n, n)
#         series_patch_num_reshaped = series_patch_num.view(batch_times_channels * heads, 1, s, s)
#
#         # 获取对应的反卷积层
#         upconv_patch_size = self.upconv_patch_size_dict[str(patch_index)]
#         upconv_patch_num = self.upconv_patch_num_dict[str(patch_index)]
#
#         # 使用反卷积和插值进行上采样，确保输出尺寸为win_size x win_size
#         series_patch_size_upsampled = upconv_patch_size(series_patch_size_reshaped)
#         series_patch_num_upsampled = upconv_patch_num(series_patch_num_reshaped)
#
#         # 验证输出尺寸是否符合预期
#         assert series_patch_size_upsampled.shape[2] == self.window_size, \
#             f"Patch-wise upsampled size mismatch: {series_patch_size_upsampled.shape[2]} vs {self.window_size}"
#         assert series_patch_num_upsampled.shape[2] == self.window_size, \
#             f"In-patch upsampled size mismatch: {series_patch_num_upsampled.shape[2]} vs {self.window_size}"
#
#         # 重塑回原始的维度结构
#         series_patch_size = series_patch_size_upsampled.view(
#             batch_times_channels, heads,
#             self.window_size,
#             self.window_size
#         )
#
#         series_patch_num = series_patch_num_upsampled.view(
#             batch_times_channels, heads,
#             self.window_size,
#             self.window_size
#         )
#
#         # 按通道进行平均，并确保输出维度为(batch_size, 1, win_size, win_size)
#         series_patch_size = reduce(series_patch_size, '(b reduce_b) h m n-> b 1 m n', 'mean', reduce_b=self.channel)
#         series_patch_num = reduce(series_patch_num, '(b reduce_b) h m n-> b 1 m n', 'mean', reduce_b=self.channel)