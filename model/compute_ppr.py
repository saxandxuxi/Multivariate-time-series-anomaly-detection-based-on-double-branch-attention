import numpy as np
import torch
from scipy.linalg import fractional_matrix_power, inv

def compute_ppr(adj_batch: torch.Tensor, alpha: float = 0.1, self_loop: bool = True) -> torch.Tensor:
    """
    批量计算Personalized PageRank (PPR) 矩阵（向量化实现）

    参数:
    adj_batch: 批量邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)
    alpha: PPR teleport probability (默认0.2)
    self_loop: 是否添加自环 (默认True)

    返回:
    ppr_batch: 批量PPR矩阵，形状为 (batch_size, num_nodes, num_nodes)
    """
    device = adj_batch.device
    batch_size, num_nodes, _ = adj_batch.shape

    # 1. 添加自环（批量操作）
    if self_loop:
        adj_batch = adj_batch + torch.eye(num_nodes, device=device).unsqueeze(0)  # (batch, n, n) + (1, n, n)

    # 2. 计算度矩阵 D^ = sum(A^, axis=2)，形状为 (batch, n, 1)
    d = adj_batch.sum(dim=2, keepdim=True)  # 每行的和，即每个节点的度数

    # 3. 计算 D^(-1/2)，避免除零：添加极小值 1e-8
    d_sqrt_inv = torch.pow(d + 1e-8, -0.5)  # (batch, n, 1)

    # 4. 对称归一化邻接矩阵 A~ = D^(-1/2) * A^ * D^(-1/2)
    #    使用广播机制进行批量矩阵乘法
    adj_normalized = d_sqrt_inv * adj_batch * d_sqrt_inv.transpose(1, 2)  # (batch, n, n)

    # 5. 计算 PPR = alpha * (I - (1-alpha)*A~)^-1
    #    生成单位矩阵 I，形状为 (1, n, n)，广播到批次维度
    eye = torch.eye(num_nodes, device=device).unsqueeze(0)  # (1, n, n)
    term = (1 - alpha) * adj_normalized  # (batch, n, n)
    inverse_matrix = torch.linalg.inv(eye - term)  # 批量矩阵求逆，要求矩阵可逆

    ppr_batch = alpha * inverse_matrix  # (batch, n, n)

    return ppr_batch


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(64, 22, 22)
    out = compute_ppr(x)
    print(out.shape)