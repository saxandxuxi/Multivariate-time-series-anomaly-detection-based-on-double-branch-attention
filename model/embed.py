import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm
import math
import pywt

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)

    # 随机生成输入张量 [64, 22, 3, 35]
    batch_size = 64
    seq_len = 22
    c_in = 3
    feature_size = 35

    # 假设输入形状为 [batch_size, seq_len, c_in]
    # 因为TokenEmbedding的forward方法中使用了x.permute(0, 2, 1)，所以输入的第三维应该是c_in
    x = torch.randn(batch_size, seq_len, feature_size)

    # 初始化DataEmbedding模型
    d_model = 256  # 设置嵌入维度等于特征大小
    model = DataEmbedding(c_in=c_in, d_model=d_model)

    # 将输入传递给模型
    output = model(x)

    # 打印输入和输出的形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")




