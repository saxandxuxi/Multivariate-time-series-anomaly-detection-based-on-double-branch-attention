import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from einops import rearrange, reduce, repeat




class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05, output_attention=False):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index,
                attn_mask):
        # Patch-wise Representation
        B, N, PatchNum, D = queries_patch_size.shape  # bs, channel, patch_num, d_model

        scale_patch_size = self.scale or 1. / torch.sqrt(torch.tensor(D, dtype=torch.float32))
        scores_patch_size = torch.einsum("bnld,bnmd->bnlm", queries_patch_size,
                                         keys_patch_size)  # bs, channel, patch_num, patch_num
        # 应用通道掩码 (bs,1,channel,channel)
        if self.mask_flag:
            # 扩展掩码维度为 [B, C, 1, PatchNum, PatchNum]
            mask_expanded = attn_mask.squeeze(1).unsqueeze(2).unsqueeze(-1)  # [B, C, 1, C, 1]
            # 重塑scores_patch_size以匹配掩码维度
            scores_expanded = scores_patch_size.unsqueeze(3)  # [B, C, PatchNum, 1, PatchNum]
            # 应用掩码（沿通道维度进行矩阵乘法）
            scores_patch_size = (scores_expanded * mask_expanded).sum(dim=3)  # [B, C, PatchNum, PatchNum]

        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))  # bs, channel, patch_num, patch_num

        # In-patch Representation
        B, N, PatchSize, D = queries_patch_num.shape  # bs, channel, patch_size, d_model

        scale_patch_num = self.scale or 1. / torch.sqrt(torch.tensor(D, dtype=torch.float32))
        scores_patch_num = torch.einsum("bnld,bnmd->bnlm", queries_patch_num,
                                        keys_patch_num)  # bs, channel, patch_size, patch_size
        # 应用通道掩码（修正：先应用掩码，再缩放，避免维度扩展）
        if attn_mask is not None:
            mask_expanded = attn_mask.squeeze(1).unsqueeze(2).unsqueeze(-1)  # [B, C, 1, C, 1]
            scores_expanded = scores_patch_num.unsqueeze(3)  # [B, C, PatchSize, 1, PatchSize]
            scores_patch_num = (scores_expanded * mask_expanded).sum(dim=3)  # [B, C, PatchSize, PatchSize]

        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1))  # bs, channel, patch_size, patch_size

        # 通道间平均
        series_patch_size = series_patch_size.mean(dim=1, keepdim=True)  # bs, 1, patch_num, patch_num
        series_patch_num = series_patch_num.mean(dim=1, keepdim=True)  # bs, 1, patch_size, patch_size

        # 上采样到window_size
        target_size = self.window_size
        series_patch_size = F.interpolate(series_patch_size, size=(target_size, target_size), mode='bilinear',
                                          align_corners=False)
        series_patch_num = F.interpolate(series_patch_num, size=(target_size, target_size), mode='bilinear',
                                         align_corners=False)

        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return None




class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads 
        
        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        
        # patch_size
        B, L, M = x_patch_size.shape##Batch*channel n d_model
        BS = B // self.channel  # 恢复真实batch_size
        N = self.channel  # 通道数
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(BS, N, L, -1)#(64,22,70,256)
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(BS, N, L, -1)

        # patch_num
        B, L, M = x_patch_num.shape#Batch*channel p d_model
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(BS, N, L, -1)
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(BS, N, L, -1)
        
        # x_ori
        B, L, d_model = x_ori.shape#(1408,70,256)
        values = self.value_projection(x_ori).view(BS, N, L, -1)
        
        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            values, patch_index,
            attn_mask
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