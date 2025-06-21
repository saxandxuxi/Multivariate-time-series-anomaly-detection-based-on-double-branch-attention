import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 在 model/DCdetector.py 中
from compute_thre import *
from metrics.metrics import combine_all_evaluation_scores


def save_energy_metrics(
        dataset,
        combined_energy,
        test_energy,
        pred: np.ndarray,
        gt: np.ndarray,
        sigma: float,
        save_dir: str = "./result"
) -> str:
    """
    保存异常分数、预测结果、真实标签到本地，并返回保存路径。

    Args:
        train_energy: 训练集异常分数 (1D array)
        test_energy: 测试集异常分数 (1D array)
        pred: 预测结果（1表示异常，0表示正常）(1D array)
        gt: 真实标签（1表示异常，0表示正常）(1D array)
        sigma: 阈值计算的sigma参数
        save_dir: 保存目录

    Returns:
        save_path: 数据保存路径
    """
    # 合并能量计算阈值

    thresh = set_global_threshold(combined_energy, sigma=sigma)  # 假设set_global_threshold已定义

    # 3. 构建保存路径（强制用项目内的result文件夹）
    os.makedirs(save_dir, exist_ok=True)
    # 文件名格式：{dataset}_energy_metrics_sigma{sigma}.npz
    save_path = os.path.join(save_dir, f"{dataset}_energy_metrics_sigma{sigma:.2f}.npz")  # sigma保留两位小数，避免文件名混乱

    # 4. 打印路径，确认位置
    print(f"数据将保存到：{save_path}")

    # 5. 检查数据非空
    assert len(combined_energy) > 0, "combined_energy为空！"
    assert len(pred) == len(gt) , "预测/标签与测试集长度不匹配！"

    # 6. 保存数据
    np.savez(
        save_path,
        combined_energy=combined_energy,
        test_energy=test_energy,
        thresh=thresh,
        pred=pred,
        gt=gt,
        sigma=sigma
    )
    return save_path


def plot_energy_distribution(save_path: str) -> None:
    """
        绘制测试集异常分数分析图：
        1. 测试集正常/异常样本的 KDE 分布（背景）
        2. 叠加预测结果（pred）与真实标签（gt）的散点：
           - TP（真异常，预测对）：绿色
           - FN（真异常，预测错）：红色
           - FP（真正常，预测错）：橙色
           - TN（真正常，预测对）：蓝色
        3. 标记异常分数阈值线
        """
    # 加载数据
    data = np.load(save_path)
    combined_energy = data["combined_energy"]
    sigma =6.0
    thresh = set_global_threshold(combined_energy, sigma=sigma)
    # thresh = np.percentile(combined_energy, 100 - 3)
    # thresh = histogram_threshold(combined_energy, bins=40)
    test_energy = data["test_energy"]
    # thresh = data["thresh"].item()  # 转换为标量
    # pred = data["pred"]
    gt = data["gt"]
    # sigma = data["sigma"].item()
    pred = (test_energy > thresh).astype(int)
    # 阈值线前后都有绿色点（TP） 是因为代码里有一段 “异常状态传播” 的逻辑 ，强行把某些样本的 pred 改成了 1，导致这些样本的 test_energy 虽然小于阈值
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    matrix = []
    scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
    for key, value in scores_simple.items():
        matrix.append(value)
        print('{0:21} : {1:0.4f}'.format(key, value))

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                               recall, f_score))
    # 2. 分类处理（KDE 分布 + 散点标记）
    ## 2.1 KDE 分布：测试集正常 vs 异常
    test_normal_energy = test_energy[gt == 0]  # 真实正常的异常分数
    test_abnormal_energy = test_energy[gt == 1]  # 真实异常的异常分数

    ## 2.2 散点标记：按 (gt, pred) 分类
    tp_mask = (gt == 1) & (pred == 1)  # 真阳性（异常预测对）
    fn_mask = (gt == 1) & (pred == 0)  # 假阴性（异常预测错）
    fp_mask = (gt == 0) & (pred == 1)  # 假阳性（正常预测错）
    tn_mask = (gt == 0) & (pred == 0)  # 真阴性（正常预测对）

    tp_energy = test_energy[tp_mask]
    fn_energy = test_energy[fn_mask]
    fp_energy = test_energy[fp_mask]
    tn_energy = test_energy[tn_mask]

    # 3. 绘图设置
    plt.figure(figsize=(14, 8))
    plt.style.use("seaborn-whitegrid")

    # 4. 绘制 KDE 分布（背景，降低透明度避免覆盖散点）
    ## 正常样本 KDE
    sns.kdeplot(
        test_normal_energy,
        label="Test (Normal, KDE)",
        fill=True,
        alpha=0.3,  # 透明度调低，突出散点
        color="#3498db"  # 蓝色
    )
    ## 异常样本 KDE
    sns.kdeplot(
        test_abnormal_energy,
        label="Test (Abnormal, KDE)",
        fill=True,
        alpha=0.3,
        color="#e74c3c"  # 红色
    )

    # 5. 绘制散点（预测 vs 真实标签）
    scatter_y = 0.01  # 固定 y 坐标，避免与 KDE 重叠（可根据实际调整）
    marker_size = 40  # 散点大小

    ## TN：真正常 + 预测正常 → 蓝色
    plt.scatter(
        tn_energy, np.full_like(tn_energy, scatter_y),
        color="#3498db", label="TN (gt=0, pred=0)",
        alpha=0.7, s=marker_size, edgecolors="white"  # 白色描边增强区分度
    )
    ## FP：真正常 + 预测异常 → 橙色
    plt.scatter(
        fp_energy, np.full_like(fp_energy, scatter_y),
        color="#f39c12", label="FP (gt=0, pred=1)",
        alpha=0.7, s=marker_size, edgecolors="white"
    )
    ## FN：真异常 + 预测正常 → 红色
    plt.scatter(
        fn_energy, np.full_like(fn_energy, scatter_y),
        color="#e74c3c", label="FN (gt=1, pred=0)",
        alpha=0.7, s=marker_size, edgecolors="white"
    )
    ## TP：真异常 + 预测异常 → 绿色
    plt.scatter(
        tp_energy, np.full_like(tp_energy, scatter_y),
        color="#27ae60", label="TP (gt=1, pred=1)",
        alpha=0.7, s=marker_size, edgecolors="white"
    )

    # 6. 标记阈值线
    plt.axvline(
        thresh, color="#9b59b6", linestyle="--", linewidth=2,
        label=f"Threshold (σ={sigma:.2f}, value={thresh:.4f})"
    )

    # 7. 美化与保存
    plt.xlabel("Anomaly Score", fontsize=14)
    plt.ylabel("Density / Scatter", fontsize=14)  # 兼容 KDE 和散点的 y 轴含义
    plt.title("Test Set Anomaly Score: Distribution + Prediction vs Ground Truth", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")  # 图例放右上角，避免遮挡
    plt.tight_layout()

    # 保存图像（路径自动拼接）
    plot_dir = os.path.dirname(save_path)
    plot_path = os.path.join(plot_dir, "energy_distribution_with_predictions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_time_series_anomaly(save_path: str) -> None:
    """
       时间序列视角可视化异常检测效果：
       1. 异常分数随时间变化的趋势（折线图）
       2. 真实异常段标记（红色背景块，体现连续性）
       3. 预测结果细分：
          - TP（绿色圆点，正确检测）
          - FN（红色叉号，漏检异常）
          - FP（橙色三角，误报异常）
       4. 阈值线标记（直观判断分数与阈值的关系）
       """
    # 1. 加载数据 & 计算预测
    data = np.load(save_path)
    combined_energy = data["combined_energy"]
    test_energy = data["test_energy"]
    gt = data["gt"].astype(int)
    sigma = 3.0 # 需与业务逻辑一致，可改为从数据加载
    thresh = set_global_threshold(combined_energy,sigma=sigma)

    # 预测后处理（与你的业务逻辑一致：标记连续异常段）
    pred = (test_energy > thresh).astype(int)
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # 向前扩展标记
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            # 向后扩展标记
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    pred = np.array(pred)

    # 2. 计算评估指标（辅助分析）
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')
    print(
        f"评估指标：\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

    # 3. 可视化设置
    plt.figure(figsize=(16, 8))
    plt.style.use("seaborn-whitegrid")
    t = np.arange(len(test_energy))  # 时间轴（假设为连续步长）

    # 4. 绘制异常分数趋势
    plt.plot(t, test_energy, label="Anomaly Score", color="darkgray", alpha=0.8, linewidth=1.5)

    # 5. 标记真实异常段（红色背景块，体现连续性）
    gt_segments = []  # 存储连续异常段的起始和结束索引
    start_idx = None
    for i in range(len(gt)):
        if gt[i] == 1 and start_idx is None:
            start_idx = i
        elif gt[i] == 0 and start_idx is not None:
            gt_segments.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:  # 处理最后一段
        gt_segments.append((start_idx, len(gt) - 1))
    # 绘制背景块
    for start, end in gt_segments:
        plt.axvspan(start, end, color="red", alpha=0.1, label="True Anomaly (Background)" if start == 0 else "")

    # 6. 标记预测结果（细分类型）
    # TP：正确检测（绿色圆点）
    tp_mask = (gt == 1) & (pred == 1)
    plt.scatter(t[tp_mask], test_energy[tp_mask], color="green", label="TP (Correct)", marker="o", s=60,
                edgecolors="white")
    # FN：漏检（红色叉号）
    fn_mask = (gt == 1) & (pred == 0)
    plt.scatter(t[fn_mask], test_energy[fn_mask], color="red", label="FN (Missed)", marker="x", s=80)
    # FP：误报（橙色三角）
    fp_mask = (gt == 0) & (pred == 1)
    plt.scatter(t[fp_mask], test_energy[fp_mask], color="orange", label="FP (False Alarm)", marker="^", s=60,
                edgecolors="white")

    # 7. 标记阈值线
    plt.axhline(thresh, color="black", linestyle="--", linewidth=2, label=f"Threshold: {thresh:.4f}")

    # 8. 美化布局
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Anomaly Score", fontsize=14)
    plt.title("Anomaly Detection Results Over Time", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()

    # 9. 保存并显示
    plot_dir = os.path.dirname(save_path)
    plot_path = os.path.join(plot_dir, "time_series_anomaly.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
if __name__ == "__main__":
    plot_energy_distribution(save_path='../result/EV24_energy_metrics_sigma3.00.npz')
    plot_time_series_anomaly(save_path='../result/EV24_energy_metrics_sigma3.00.npz')


