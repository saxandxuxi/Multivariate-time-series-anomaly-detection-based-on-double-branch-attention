import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self, num_layers=2, dropout=0.1):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, adj, out_ft=64, mask=None):
        """
        Args:
            feat: 输入特征，维度 (B, M, T, D)
                  B: 批次大小，M: 通道数，T: 时间点数，D: 特征维度
            adj: 邻接矩阵，维度 (B, 1, M, M)
                  表示通道间的相似性（由马氏距离或PPR生成）
            out_ft: 输出特征维度，默认64
            mask: 可选掩码（未使用，保留接口兼容）
        Returns:
            h: 各层输出特征，维度 (B, M, T, out_ft)
            hg: 图级表示，维度 (B, T, num_layers*out_ft)
        """
        B, M, T, D = feat.shape

        # 如果是第一次前向传播，初始化网络层
        if len(self.layers) == 0:
            self.layers.append(nn.Linear(D, out_ft))
            for _ in range(self.num_layers - 1):
                self.layers.append(nn.Linear(out_ft, out_ft))

            # 为每个线性层创建对应的偏置参数
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)

        # 重塑为 (B*T, M, D) 以便按时间点处理通道间关系
        feat = feat.permute(0, 2, 1, 3).reshape(B * T, M, D)  # (B*T, M, D)

        # 扩展邻接矩阵到每个时间点
        adj = adj.expand(-1, T, -1, -1).reshape(B * T, M, M)  # (B*T, M, M)

        h = self.layers[0](feat)  # 初始特征变换
        h = self.act(h + self.bias)  # 添加偏置并激活
        # h = self.dropout(h)

        # 图卷积：通道间消息传递
        hg = torch.mean(h, dim=1, keepdim=True)  # 初始图级表示 (B*T, 1, out_ft)

        for idx in range(1, self.num_layers):
            h = torch.bmm(adj, h)  # 通道间消息传递
            h = self.layers[idx](h)
            h = self.act(h + self.bias)
            # h = self.dropout(h)

            # 拼接多层图级表示
            hg = torch.cat([hg, torch.mean(h, dim=1, keepdim=True)], dim=-1)

        # 恢复维度：(B*T, M, out_ft) → (B, M, T, out_ft)
        h = h.reshape(B, T, M, -1).permute(0, 2, 1, 3)
        # 图级表示：(B*T, 1, num_layers*out_ft) → (B, T, num_layers*out_ft)
        hg = hg.squeeze(1).reshape(B, T, -1)

        return h


if __name__ == '__main__':
    feat = torch.randn(64, 22, 3, 3)
    adj = torch.randn(64, 1, 22, 22)
    model = GCN(num_layers=2, dropout=0.1)
    out = model(feat, adj,out_ft=3)
    print(out[0].shape)