import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicReconHead(nn.Module):
    """动态重建头，能够根据输入维度自适应调整"""

    def __init__(self, hidden_dim_ratio=0.5):
        super(DynamicReconHead, self).__init__()
        self.hidden_dim_ratio = hidden_dim_ratio

    def forward(self, x, target_dim):
        """
        x: 输入张量 (bs, channel, patch_size*patch_size)
        target_dim: 目标输出维度 (通常为通道数)
        """
        bs, ch, input_dim = x.shape
        hidden_dim = int(input_dim * self.hidden_dim_ratio)

        # 动态创建并应用MLP
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim)
        ).to(x.device)

        return mlp(x)



if __name__ == '__main__':
    model = DynamicReconHead()

    x = torch.randn(64, 22, 10, 10)
    bs, ch, pn, _ = x.shape
    x = x.reshape(bs, ch, -1)
    out = model(x, target_dim=105)

    print(out.shape)
