import math
from tkinter import _flatten

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ray.experimental import channel

from .DynamicReconHead import DynamicReconHead
from .GCN_representation import GCN
from .Mahalanobis_mask import Mahalanobis_mask, PyramidMahalanobisMask, MultiDistanceChannelClustering
from .TCM import Linear_extractor_cluster
from .attn import DAC_structure, AttentionLayer
from .compute_ppr import compute_ppr
from .embed import DataEmbedding, TokenEmbedding
from .RevIN import RevIN

class GCdetector(nn.Module):
    def __init__(self, win_size, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=[3, 5, 7], channel=55,
                 d_ff=512, dropout=0.0, activation='gelu', output_attention=True,num_experts = 3,k=2):
        super(GCdetector, self).__init__()
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
        self.cross_channel_ffn =Mahalanobis_mask(win_size)

        # Dual Attention Encoder
        self.encoder = GCN(in_ft=d_model,out_ft=d_model)

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        B, L, M = x.shape  # Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []
        series_rec_mean = []
        revin_layer = RevIN(num_features=M)
        # Instance Normalization Operation
        x = revin_layer(x, 'norm')
        x_channel = x.permute(0, 2, 1)
        channel_mask = self.cross_channel_ffn(x_channel)#(bs,1,22,22)
        self.recon_head = DynamicReconHead()
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
            # (B,M,3,256) ,(B, M, num_layers*out_ft))
            n_series,g_series = self.encoder(x_patch_num,channel_mask)
            n_prior,g_prior = self.prior_centers(x_patch_size,channel_mask)
            #patch-wise recon
            patch_wise_recon = self.recon_head(n_series,self.win_size)
            # upsample

            series_patch_mean.append(g_series), prior_patch_mean.append(g_prior)
            series_rec_mean.append(patch_wise_recon)

        series_patch_mean = list(_flatten(series_patch_mean))  ##[B,1,L,L]
        prior_patch_mean = list(_flatten(prior_patch_mean))
        series_rec_mean = list(_flatten(series_rec_mean))
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
            return series_patch_mean, prior_patch_mean, series_features, prior_features, self.series_centers, self.prior_centers,series_rec_mean
        else:
            return None