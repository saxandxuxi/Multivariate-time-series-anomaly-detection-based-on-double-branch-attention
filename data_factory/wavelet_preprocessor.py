import numpy as np
import pywt
import os
import argparse
from tqdm import tqdm


def wavelet_denoising(data, wavelet='db4', level=1, eps=1e-8):
    """
    对输入数据进行小波去噪处理
    参数:
        data: numpy数组，形状为 [样本数, 特征维度]
        wavelet: 小波函数名称
        level: 分解级别
        eps: 防止除零的小常数
    返回:
        去噪后的数据
    """
    denoised_data = np.zeros_like(data)
    num_samples, feature_dim = data.shape

    # 假设每个特征是一个完整的时间序列
    for i in tqdm(range(num_samples), desc="Processing samples"):
        # 提取第i个样本的所有特征（每个特征作为一个时间序列）
        sample = data[i, :]  # 形状: [特征维度]

        try:
            # 对整个样本的所有特征一起进行小波变换
            coeffs = pywt.wavedec(sample, wavelet, level=level, axis=0)

            # 阈值处理
            std = np.std(sample, axis=0, keepdims=True)
            std = np.maximum(std, eps)  # 防止除零
            threshold = 0.5 * std

            coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

            # 重构信号
            reconstructed = pywt.waverec(coeffs, wavelet, axis=0)

            # 确保长度匹配
            if len(reconstructed) > feature_dim:
                reconstructed = reconstructed[:feature_dim]
            elif len(reconstructed) < feature_dim:
                padded = np.zeros_like(sample)
                padded[:len(reconstructed)] = reconstructed
                reconstructed = padded

            denoised_data[i, :] = reconstructed

        except Exception as e:
            print(f"小波变换失败，样本 {i}: {e}")
            denoised_data[i, :] = sample  # 失败时返回原始样本

    return denoised_data


def main(args):
    # 确保保存目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载原始数据
    print(f"Loading data from {args.data_path}...")
    train_path = os.path.join(args.data_path, f"{args.dataset}_train.npy")
    test_path = os.path.join(args.data_path, f"{args.dataset}_test.npy")

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # 应用小波变换
    print(f"Applying wavelet transform (wavelet={args.wavelet}, level={args.level})...")
    train_denoised = wavelet_denoising(train_data, args.wavelet, args.level)
    test_denoised = wavelet_denoising(test_data, args.wavelet, args.level)

    # 保存处理后的数据
    train_output_path = os.path.join(args.output_dir, f"{args.dataset}_train_wavelet.npy")
    test_output_path = os.path.join(args.output_dir, f"{args.dataset}_test_wavelet.npy")

    np.save(train_output_path, train_denoised)
    np.save(test_output_path, test_denoised)

    print(f"Wavelet-processed data saved to:")
    print(f"  Train: {train_output_path}")
    print(f"  Test: {test_output_path}")


if __name__ == "__main__":
    class Args:
        data_path = '../dataset/EV47'  # 原始数据路径
        dataset = 'EV47'  # 数据集名称
        output_dir = '../pro_data/EV47'  # 输出路径
        wavelet = 'db4'  # 小波类型
        level = 1  # 分解级别
        copy_labels = True  # 是否复制标签


    args = Args()
    main(args)