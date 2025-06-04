import numpy as np
import torch

def set_global_threshold(combined_energy, sigma=1.0):#47:1
    mean = np.mean(combined_energy)
    std = np.std(combined_energy)
    threshold = mean + sigma * std
    return threshold

def set_dynamic_percentile_threshold(combined_energy, percentile=95):
    # 按时间步（列）计算百分位数，输出形状为 [序列长度]
    threshold = np.percentile(combined_energy, 100 - percentile, axis=0)
    return threshold

def iqr_threshold(combined_energy, factor=1.5):
    """基于四分位数间距(IQR)计算阈值"""
    scores = combined_energy.flatten()
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    return q3 + factor * iqr  # 上界阈值，超过此值视为异常
def mad_threshold(combined_energy, factor=3.0):
    """基于中位数绝对偏差(MAD)计算阈值"""
    scores = combined_energy.flatten()
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return median + factor * mad * 1.4826  # 1.4826因子使MAD等价于标准差


def fast_lof_threshold(combined_energy, radius=0.1, contamination=0.05):
    """快速局部离群因子阈值计算（兼容PyTorch和NumPy）"""
    # 判断输入类型
    if isinstance(combined_energy, torch.Tensor):
        scores = combined_energy.flatten().unsqueeze(1)
        is_tensor = True
    else:
        scores = combined_energy.flatten().reshape(-1, 1)
        is_tensor = False

    # 计算每个点的局部可达密度(简化版)
    lrd = torch.zeros(len(scores)) if is_tensor else np.zeros(len(scores))

    for i, x in enumerate(scores):
        # 计算半径内的邻居数（根据类型选择操作）
        if is_tensor:
            neighbors = torch.sum(torch.abs(scores - x) <= radius)
        else:
            neighbors = np.sum(np.abs(scores - x) <= radius)
        lrd[i] = 1.0 / (neighbors + 1e-10)  # 避免除零

    # 计算LOF分数(局部离群因子)
    lof_scores = torch.zeros(len(scores)) if is_tensor else np.zeros(len(scores))

    for i, x in enumerate(scores):
        if is_tensor:
            neighbors = torch.where(torch.abs(scores - x) <= radius)[0]
        else:
            neighbors = np.where(np.abs(scores - x) <= radius)[0]

        if len(neighbors) > 0:
            lof_scores[i] = torch.mean(lrd[neighbors]) / lrd[i] if is_tensor else np.mean(lrd[neighbors]) / lrd[i]

    # 转换为NumPy数组进行百分位数计算
    if is_tensor:
        lof_scores = lof_scores.numpy()

    # 返回前contamination比例的高分值作为阈值
    threshold = np.percentile(lof_scores, (1 - contamination) * 100)
    return threshold


def histogram_threshold(combined_energy, bins=50):
    """基于直方图的自适应阈值计算"""
    scores = combined_energy.flatten()
    hist, bin_edges = np.histogram(scores, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # 找到直方图的第一个谷值作为阈值
    for i in range(1, len(hist) - 1):
        if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
            return bin_centers[i]

    # 如果没有找到谷值，返回95%分位数
    return np.percentile(scores, 95)


def sliding_window_threshold(combined_energy, window_size=5, sigma=2.0):
    """简化的滑动窗口阈值计算（兼容PyTorch和NumPy）"""
    import torch

    # 判断输入类型
    if isinstance(combined_energy, torch.Tensor):
        scores = combined_energy.flatten()
        thresholds = torch.zeros_like(scores)
        is_tensor = True
    else:
        scores = combined_energy.flatten()
        thresholds = np.zeros_like(scores)
        is_tensor = False

    for i in range(len(scores)):
        start = max(0, i - window_size)
        window = scores[start:i + 1]

        # 根据类型选择对应的统计函数
        if is_tensor:
            mean_val = torch.mean(window)
            std_val = torch.std(window)
        else:
            mean_val = np.mean(window)
            std_val = np.std(window)

        thresholds[i] = mean_val + sigma * std_val

    # 返回全局最大阈值
    if is_tensor:
        return thresholds.max().item()
    else:
        return np.max(thresholds) # 或返回最后一个阈值

if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    x= torch.randn(64,70)
    th1=iqr_threshold(x)
    th2=fast_lof_threshold(x)
    th2=mad_threshold(x)
    th3=histogram_threshold(x)
    th4=sliding_window_threshold(x)
    print(th1, th2, th3, th4)








