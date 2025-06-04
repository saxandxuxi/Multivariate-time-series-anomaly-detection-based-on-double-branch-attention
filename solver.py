import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from sklearn.metrics import precision_recall_fscore_support
from torch.optim import Adam

from compute_thre import sliding_window_threshold,set_global_threshold
from utils.utils import *
from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        """保存模型权重和中心向量"""
        state_dict = {
            'model_state_dict': model.state_dict(),
            'series_centers': [center.detach().cpu() for center in model.series_centers],
            'prior_centers': [center.detach().cpu() for center in model.prior_centers],
            # 可添加其他需要保存的参数
        }
        # torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        torch.save(state_dict, os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.prior_centers = None
        self.series_centers = None
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.build_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 新增：初始化对比学习相关参数
        self.use_contrastive = config.get('use_contrastive', True)
        # self.contrastive_weight = config.get('contrastive_weight', 0.15)#EV24:0.15 EV47:0.15
        self.temperature = config.get('temperature', 0.25)

        # 新增：初始化投影头（使用正确的特征维度 `win_size`）
        if self.use_contrastive:
            feature_dim = self.win_size  # 直接使用 `win_size` 作为特征维度
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.GELU()
            ).to(self.device)

        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads,
                                d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size,
                                channel=self.input_c,dropout=self.dropout)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)#增加了weight_decay=0.0001

    def _compute_KL_losses(self, series, prior):
        """辅助函数计算series_loss和prior_loss"""
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            # 计算 L_P = KL(P, Stopgrad(N)) + KL(Stopgrad(N), P)
            n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                  self.win_size)
            n_stopgrad = n_normalized.detach()  # 固定N的梯度
            kl_p1 = my_kl_loss(series[u], n_stopgrad)  # 修正：使用series[u]作为P
            kl_p2 = my_kl_loss(n_stopgrad, series[u])  # 修正：使用series[u]作为P
            series_loss += torch.mean(kl_p1 + kl_p2)  # 双向求和,补丁内

            # 计算 L_N = KL(N, Stopgrad(P)) + KL(Stopgrad(P), N)
            p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
            p_stopgrad = p_normalized.detach()  # 固定P的梯度
            kl_n1 = my_kl_loss(prior[u], p_stopgrad)  # 修正：使用prior[u]作为N
            kl_n2 = my_kl_loss(p_stopgrad, prior[u])  # 修正：使用prior[u]作为N
            prior_loss += torch.mean(kl_n1 + kl_n2)  # 双向求和，补丁间
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        prior_loss = prior_loss * self.contrastive_weight
        return series_loss, prior_loss

    def compute_center_loss(self, series_features, prior_features):
        """计算中心向量损失，返回维度为 [B, L] 的损失矩阵"""
        batch_size, seq_len = series_features[0].shape
        total_loss = torch.zeros(batch_size, seq_len, device=series_features[0].device)

        for i in range(len(self.series_centers)):
            series_diff = series_features[i] - self.series_centers[i]  # [B, L]
            prior_diff = prior_features[i] - self.prior_centers[i]  # [B, L]

            # 计算 L2 距离并扩展为 [B, L]
            series_dist = torch.norm(series_diff, p=2, dim=1, keepdim=True).expand(-1, seq_len)
            prior_dist = torch.norm(prior_diff, p=2, dim=1, keepdim=True).expand(-1, seq_len)

            total_loss += (series_dist + prior_dist) / 2

        return total_loss / len(self.series_centers)  # [B, L]

    def _compute_contrastive_loss(self, series, prior):
        """计算对比学习损失，保持[B,L]维度输出"""
        batch_size = series[0].size(0)
        win_size = series[0].size(3)  # 假设特征维度由 win_size 决定
        seq_len = series[0].size(2)  # 序列长度 L
        contrastive_loss = torch.zeros(batch_size, seq_len, device=self.device)
        num_scales = len(self.patch_size)
        num_layers = len(series) // num_scales

        def feature_perturbation(x):
            """针对特征表示的扰动函数"""
            mask = torch.rand_like(x) > 0.3
            masked = x * mask.float()
            if seq_len > 1:
                shift_val = torch.randint(-seq_len // 8, seq_len // 8, (1,), device=x.device)[0]
                shift = int(shift_val)
                if shift == 0:
                    shift = 1 if torch.rand(1) > 0.5 else -1
                shifted = torch.roll(x, shifts=shift, dims=2)
            else:
                shifted = x
            return 0.5 * masked + 0.5 * shifted

        for scale_idx in range(num_scales):
            for layer_idx in range(num_layers):
                u = scale_idx * num_layers + layer_idx
                s = series[u]  # [B, 1, L, L]
                p = prior[u]  # [B, 1, L, L]

                s_neg = feature_perturbation(s)
                p_neg = feature_perturbation(p)

                # 修正展平方式，将 [B, 1, L, L] 转为 [B, L, L]
                s_flat = s.reshape(batch_size, seq_len, seq_len)  # [B, L, L]
                p_flat = p.reshape(batch_size, seq_len, seq_len)
                s_neg_flat = s_neg.reshape(batch_size, seq_len, seq_len)
                p_neg_flat = p_neg.reshape(batch_size, seq_len, seq_len)

                s_proj = torch.stack([self.projection(s_flat[:, i, :]) for i in range(seq_len)], dim=1)
                p_proj = torch.stack([self.projection(p_flat[:, i, :]) for i in range(seq_len)], dim=1)
                s_neg_proj = torch.stack([self.projection(s_neg_flat[:, i, :]) for i in range(seq_len)], dim=1)
                p_neg_proj = torch.stack([self.projection(p_neg_flat[:, i, :]) for i in range(seq_len)], dim=1)

                s_norm = F.normalize(s_proj, p=2, dim=2)
                p_norm = F.normalize(p_proj, p=2, dim=2)
                s_neg_norm = F.normalize(s_neg_proj, p=2, dim=2)
                p_neg_norm = F.normalize(p_neg_proj, p=2, dim=2)

                pos_sim = torch.sum(s_norm * p_norm, dim=2) / self.temperature  # [B,L]
                neg_sim1 = torch.sum(s_norm * p_neg_norm, dim=2) / self.temperature  # [B,L]
                neg_sim2 = torch.sum(s_neg_norm * p_norm, dim=2) / self.temperature  # [B,L]

                time_loss = -torch.log(
                    torch.exp(pos_sim) /
                    (torch.exp(pos_sim) + torch.exp(neg_sim1) + torch.exp(neg_sim2) + 1e-8)
                )  # [B,L]

                contrastive_loss += time_loss  # 显式聚合为标量

        return contrastive_loss / (num_scales * num_layers)

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            # series, prior = self.model(input)
            series, prior,series_features, prior_features, self.series_centers,self.prior_centers = self.model(input)
            series_loss, prior_loss = self._compute_KL_losses(series, prior)
            center_loss = self.compute_center_loss(series_features, prior_features)
            center_loss = torch.mean(center_loss)
            loss = abs(prior_loss - series_loss) + center_loss
            loss_1.append(loss.item())
            # loss_1.append((prior_loss - series_loss).item())
        return np.average(loss_1), np.average(loss_2)

    def train(self):
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)

        # # 定义两个可学习的权重参数
        # series_weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # prior_weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # optimizer_weight = Adam([series_weight, prior_weight], lr=self.lr)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                # series, prior = self.model(input)
                # series_loss, prior_loss = self._compute_KL_losses(series, prior)
                # loss = prior_loss - series_loss
                series, prior, series_features, prior_features, self.series_centers, self.prior_centers = self.model(input)
                series_loss, prior_loss = self._compute_KL_losses(series, prior)
                center_loss = self.compute_center_loss(series_features, prior_features)
                center_loss = torch.mean(center_loss)
                loss = abs(prior_loss - series_loss) + center_loss
                if (i + 1) % 100 == 0:
                    # speed = (time.time() - time_now) / iter_count
                    # left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tEpoch: {:d};Loss: {:.4f}'.format(
                        epoch + 1,  # 当前epoch（从1开始）
                        loss.item()
                    ))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def _compute_attens_energy(self, data_loader, temperature):
        """辅助函数计算能量值"""
        attens_energy = []
        for i, (input_data, labels) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            series, prior,series_features, prior_features,_,_ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                      self.win_size)
                p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)
                series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
                prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature
            center_loss = self.compute_center_loss(series_features, prior_features)
            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            metric = torch.softmax((-series_loss - prior_loss-center_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        return np.concatenate(attens_energy, axis=0).reshape(-1)

    def _compute_attens_energy_and_labels(self, data_loader, temperature):
        """计算给定数据加载器的注意力能量和标签"""
        attens_energy = []
        labels_list = []
        for i, (input_data, labels) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            series, prior,series_features, prior_features,_,_= self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                      self.win_size)
                p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)
                series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
                prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature

            center_loss = self.compute_center_loss(series_features, prior_features)
            metric = torch.softmax((-series_loss - prior_loss - center_loss), dim=-1)
            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            labels_list.append(labels)
        return np.concatenate(attens_energy, axis=0).reshape(-1), np.concatenate(labels_list, axis=0).reshape(-1)

    def test(self):
        model_path = os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.trainning=False
        # 加载中心向量并移至设备
        device = next(self.model.parameters()).device
        for i, (s_center, p_center) in enumerate(zip(checkpoint['series_centers'], checkpoint['prior_centers'])):
            self.model.series_centers[i] = nn.Parameter(s_center.to(device))
            self.model.prior_centers[i] = nn.Parameter(p_center.to(device))
        print(f"模型和中心向量已从 {model_path} 加载")
        temperature = 50
        self.prior_centers = self.model.series_centers
        self.series_centers = self.model.prior_centers
        # (1) stastic on the train set
        train_energy = self._compute_attens_energy(self.train_loader, temperature)
        # (2) find the threshold
        test_energy = self._compute_attens_energy(self.thre_loader, temperature)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        print('==============================测试阶段训练完成============================')
        # thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        thresh = set_global_threshold(combined_energy,sigma=self.sigma)
        # thresh = sliding_window_threshold(combined_energy)
        print("Threshold :", thresh)
        # (3)使用辅助函数计算thre_loader的能量和标签（用于评估）
        test_energy, test_labels = self._compute_attens_energy_and_labels(self.thre_loader, temperature)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

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

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))


        return accuracy, precision, recall, f_score

    # def test(self):
    #     self.model.load_state_dict(
    #         torch.load(
    #             os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
    #     self.model.eval()
    #     temperature = 50
    #
    #     # (1) stastic on the train set
    #     attens_energy = []
    #     for i, (input_data, labels) in enumerate(self.train_loader):
    #         input = input_data.float().to(self.device)
    #         series, prior = self.model(input)
    #         series_loss = 0.0
    #         prior_loss = 0.0
    #         for u in range(len(prior)):
    #             # if u == 0:
    #             #     series_loss = my_kl_loss(series[u], (
    #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                    self.win_size)).detach()) * temperature
    #             #     prior_loss = my_kl_loss(
    #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                 self.win_size)),
    #             #         series[u].detach()) * temperature
    #             # else:
    #             #     series_loss += my_kl_loss(series[u], (
    #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                    self.win_size)).detach()) * temperature
    #             #     prior_loss += my_kl_loss(
    #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                 self.win_size)),
    #             #         series[u].detach()) * temperature
    #             # 获取两个分支的表示
    #             P = series[u]
    #             N = prior[u]
    #             # 计算P到N的KL散度（N停止梯度），保留原始归一化操作
    #             N_normalized = N / torch.unsqueeze(torch.sum(N, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
    #             kl_p_to_n = my_kl_loss(P, N_normalized.detach())  # 对应原代码中的series_loss
    #
    #             # 计算N到P的KL散度（P停止梯度），保留原始归一化操作
    #             P_normalized = P / torch.unsqueeze(torch.sum(P, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
    #             kl_n_to_p = my_kl_loss(N, P_normalized.detach())  # 对应原代码中的prior_loss
    #
    #             # 原代码逻辑：根据u是否为0决定初始化或累加
    #             if u == 0:
    #                 series_loss = kl_p_to_n
    #                 prior_loss = kl_n_to_p
    #             else:
    #                 series_loss += kl_p_to_n
    #                 prior_loss += kl_n_to_p
    #
    #         # 应用温度系数（移到循环外，避免重复缩放）
    #         series_loss = series_loss * temperature
    #         prior_loss = prior_loss * temperature
    #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    #         cri = metric.detach().cpu().numpy()
    #         attens_energy.append(cri)
    #
    #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    #     train_energy = np.array(attens_energy)
    #
    #     # (2) find the threshold
    #     attens_energy = []
    #     for i, (input_data, labels) in enumerate(self.thre_loader):
    #         input = input_data.float().to(self.device)
    #         series, prior = self.model(input)
    #         series_loss = 0.0
    #         prior_loss = 0.0
    #         for u in range(len(prior)):
    #             # if u == 0:
    #             #     series_loss = my_kl_loss(series[u], (
    #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                    self.win_size)).detach()) * temperature
    #             #     prior_loss = my_kl_loss(
    #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                 self.win_size)),
    #             #         series[u].detach()) * temperature
    #             # else:
    #             #     series_loss += my_kl_loss(series[u], (
    #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                    self.win_size)).detach()) * temperature
    #             #     prior_loss += my_kl_loss(
    #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                 self.win_size)),
    #             #         series[u].detach()) * temperature
    #             # 获取两个分支的表示
    #             P = series[u]
    #             N = prior[u]
    #             # 计算P到N的KL散度（N停止梯度），保留原始归一化操作
    #             N_normalized = N / torch.unsqueeze(torch.sum(N, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
    #             kl_p_to_n = my_kl_loss(P, N_normalized.detach())  # 对应原代码中的series_loss
    #
    #             # 计算N到P的KL散度（P停止梯度），保留原始归一化操作
    #             P_normalized = P / torch.unsqueeze(torch.sum(P, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
    #             kl_n_to_p = my_kl_loss(N, P_normalized.detach())  # 对应原代码中的prior_loss
    #
    #             # 原代码逻辑：根据u是否为0决定初始化或累加
    #             if u == 0:
    #                 series_loss = kl_p_to_n
    #                 prior_loss = kl_n_to_p
    #             else:
    #                 series_loss += kl_p_to_n
    #                 prior_loss += kl_n_to_p
    #
    #                 # 应用温度系数（移到循环外，避免重复缩放）
    #         series_loss = series_loss * temperature
    #         prior_loss = prior_loss * temperature
    #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    #         cri = metric.detach().cpu().numpy()
    #         attens_energy.append(cri)
    #
    #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    #     test_energy = np.array(attens_energy)
    #     combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    #     thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
    #     print("Threshold :", thresh)
    #
    #     # (3) evaluation on the test set
    #     test_labels = []
    #     attens_energy = []
    #     for i, (input_data, labels) in enumerate(self.thre_loader):
    #         input = input_data.float().to(self.device)
    #         series, prior = self.model(input)
    #         series_loss = 0.0
    #         prior_loss = 0.0
    #         for u in range(len(prior)):
    #             # if u == 0:
    #             #     series_loss = my_kl_loss(series[u], (
    #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                    self.win_size)).detach()) * temperature
    #             #     prior_loss = my_kl_loss(
    #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                 self.win_size)),
    #             #         series[u].detach()) * temperature
    #             # else:
    #             #     series_loss += my_kl_loss(series[u], (
    #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                    self.win_size)).detach()) * temperature
    #             #     prior_loss += my_kl_loss(
    #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #             #                                                                                 self.win_size)),
    #             #         series[u].detach()) * temperature
    #             # 获取两个分支的表示
    #             P = series[u]
    #             N = prior[u]
    #             # 计算P到N的KL散度（N停止梯度），保留原始归一化操作
    #             N_normalized = N / torch.unsqueeze(torch.sum(N, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
    #             kl_p_to_n = my_kl_loss(P, N_normalized.detach())  # 对应原代码中的series_loss
    #
    #             # 计算N到P的KL散度（P停止梯度），保留原始归一化操作
    #             P_normalized = P / torch.unsqueeze(torch.sum(P, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
    #             kl_n_to_p = my_kl_loss(N, P_normalized.detach())  # 对应原代码中的prior_loss
    #
    #             # 原代码逻辑：根据u是否为0决定初始化或累加
    #             # 原代码逻辑：根据u是否为0决定初始化或累加
    #             if u == 0:
    #                 series_loss = kl_p_to_n
    #                 prior_loss = kl_n_to_p
    #             else:
    #                 series_loss += kl_p_to_n
    #                 prior_loss += kl_n_to_p
    #
    #                 # 应用温度系数（移到循环外，避免重复缩放）
    #         series_loss = series_loss * temperature
    #         prior_loss = prior_loss * temperature
    #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    #         cri = metric.detach().cpu().numpy()
    #         attens_energy.append(cri)
    #         test_labels.append(labels)
    #
    #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    #     test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    #     test_energy = np.array(attens_energy)
    #     test_labels = np.array(test_labels)
    #
    #     pred = (test_energy > thresh).astype(int)
    #     gt = test_labels.astype(int)
    #
    #     matrix = [self.index]
    #     scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
    #     for key, value in scores_simple.items():
    #         matrix.append(value)
    #         print('{0:21} : {1:0.4f}'.format(key, value))
    #
    #     anomaly_state = False
    #     for i in range(len(gt)):
    #         if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
    #             anomaly_state = True
    #             for j in range(i, 0, -1):
    #                 if gt[j] == 0:
    #                     break
    #                 else:
    #                     if pred[j] == 0:
    #                         pred[j] = 1
    #             for j in range(i, len(gt)):
    #                 if gt[j] == 0:
    #                     break
    #                 else:
    #                     if pred[j] == 0:
    #                         pred[j] = 1
    #         elif gt[i] == 0:
    #             anomaly_state = False
    #         if anomaly_state:
    #             pred[i] = 1
    #
    #     pred = np.array(pred)
    #     gt = np.array(gt)
    #
    #     from sklearn.metrics import precision_recall_fscore_support
    #     from sklearn.metrics import accuracy_score
    #
    #     accuracy = accuracy_score(gt, pred)
    #     precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    #     print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
    #
    #     if self.data_path == 'UCR' or 'UCR_AUG':
    #         import csv
    #         with open('result/'+self.data_path+'.csv', 'a+') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(matrix)
    #
    #     return accuracy, precision, recall, f_score

# class Solver(object):
#     DEFAULTS = {}
#
#     def __init__(self, config):
#
#         self.__dict__.update(Solver.DEFAULTS, **config)
#
#         self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
#                                                win_size=self.win_size, mode='train', dataset=self.dataset, )
#         self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
#                                               win_size=self.win_size, mode='val', dataset=self.dataset)
#         self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
#                                               win_size=self.win_size, mode='test', dataset=self.dataset)
#         self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
#                                               win_size=self.win_size, mode='thre', dataset=self.dataset)
#
#         self.build_model()
#
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         if self.loss_fuc == 'MAE':
#             self.criterion = nn.L1Loss()
#         elif self.loss_fuc == 'MSE':
#             self.criterion = nn.MSELoss()
#
#     def build_model(self):
#         self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads,
#                                 d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size,
#                                 channel=self.input_c)
#
#         if torch.cuda.is_available():
#             self.model.cuda()
#
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#
#     def vali(self, vali_loader):
#         self.model.eval()
#         loss_1 = []
#         loss_2 = []
#         for i, (input_data, _) in enumerate(vali_loader):
#             input = input_data.float().to(self.device)
#             series, prior = self.model(input)
#             series_loss = 0.0
#             prior_loss = 0.0
#             for u in range(len(prior)):
#                 # series_loss += (torch.mean(my_kl_loss(series[u], (
#                 #         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                 #                                                                                self.win_size)).detach())) + torch.mean(
#                 #     my_kl_loss(
#                 #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                 #                                                                                 self.win_size)).detach(),
#                 #         series[u])))
#                 # prior_loss += (torch.mean(
#                 #     my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                 #                                                                                        self.win_size)),
#                 #                series[u].detach())) + torch.mean(
#                 #     my_kl_loss(series[u].detach(),
#                 #                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                 #                                                                                        self.win_size)))))
#                 # # 归一化prior表示（保持原始逻辑）
#                 # n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                 #                                                                                       self.win_size)
#                 # n_stopgrad = n_normalized.detach()  # 固定N的梯度
#                 # kl_p1 = my_kl_loss(prior[u], n_stopgrad)  # KL(P || Stopgrad(N))
#                 # kl_p2 = my_kl_loss(n_stopgrad, prior[u])  # KL(Stopgrad(N) || P)
#                 # series_loss += torch.mean(kl_p1 + kl_p2)  # 双向求和
#                 #
#                 # # 计算 L_N = KL(N, Stopgrad(P)) + KL(Stopgrad(P), N)
#                 # p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                 #                                                                                         self.win_size)
#                 # p_stopgrad = p_normalized.detach()  # 固定P的梯度
#                 # kl_n1 = my_kl_loss(series[u], p_stopgrad)  # KL(N || Stopgrad(P))
#                 # kl_n2 = my_kl_loss(p_stopgrad, series[u])  # KL(Stopgrad(P) || N)
#                 # prior_loss += torch.mean(kl_n1 + kl_n2)  # 双向求和
#
#                 n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                       self.win_size)
#                 n_stopgrad = n_normalized.detach()  # 固定N的梯度
#                 kl_p1 = my_kl_loss(series[u], n_stopgrad)  # 修正：使用series[u]作为P
#                 kl_p2 = my_kl_loss(n_stopgrad, series[u])  # 修正：使用series[u]作为P
#                 series_loss += torch.mean(kl_p1 + kl_p2)  # 双向求和
#
#                 # 计算 L_N = KL(N, Stopgrad(P)) + KL(Stopgrad(P), N)
#                 p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                         self.win_size)
#                 p_stopgrad = p_normalized.detach()  # 固定P的梯度
#                 kl_n1 = my_kl_loss(prior[u], p_stopgrad)  # 修正：使用prior[u]作为N
#                 kl_n2 = my_kl_loss(p_stopgrad, prior[u])  # 修正：使用prior[u]作为N
#                 prior_loss += torch.mean(kl_n1 + kl_n2)  # 双向求和
#
#             series_loss = series_loss / len(prior)
#             prior_loss = prior_loss / len(prior)
#
#             loss_1.append((prior_loss - series_loss).item())
#
#         return np.average(loss_1), np.average(loss_2)
#
#     def train(self):
#
#         time_now = time.time()
#         path = self.model_save_path
#         if not os.path.exists(path):
#             os.makedirs(path)
#         early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
#         train_steps = len(self.train_loader)
#
#         # # 定义两个可学习的权重参数
#         # series_weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
#         # prior_weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))
#         # optimizer_weight = Adam([series_weight, prior_weight], lr=self.lr)
#
#         for epoch in range(self.num_epochs):
#             iter_count = 0
#
#             epoch_time = time.time()
#             self.model.train()
#             for i, (input_data, labels) in enumerate(self.train_loader):
#
#                 self.optimizer.zero_grad()
#                 iter_count += 1
#                 input = input_data.float().to(self.device)
#                 series, prior = self.model(input)
#                 # in-patch 表示:series patch_num:prior
#                 series_loss = 0.0
#                 prior_loss = 0.0
#
#                 for u in range(len(prior)):
#                     # series_loss += (torch.mean(my_kl_loss(series[u], (
#                     #         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                     #                                                                                self.win_size)).detach())) + torch.mean(
#                     #     my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                     #                                                                                        self.win_size)).detach(),
#                     #                series[u])))
#                     # prior_loss += (torch.mean(my_kl_loss(
#                     #     (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                     #                                                                             self.win_size)),
#                     #     series[u].detach())) + torch.mean(
#                     #     my_kl_loss(series[u].detach(), (
#                     #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                     #                                                                                    self.win_size)))))
#                     # n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                     #                                                                                       self.win_size)
#                     # n_stopgrad = n_normalized.detach()  # 固定N的梯度
#                     # kl_p1 = my_kl_loss(prior[u], n_stopgrad)  # KL(P || Stopgrad(N))
#                     # kl_p2 = my_kl_loss(n_stopgrad, prior[u])  # KL(Stopgrad(N) || P)
#                     # series_loss += torch.mean(kl_p1 + kl_p2)  # 双向求和
#                     #
#                     # # 计算 L_N = KL(N, Stopgrad(P)) + KL(Stopgrad(P), N)
#                     # p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                     #                                                                                         self.win_size)
#                     # p_stopgrad = p_normalized.detach()  # 固定P的梯度
#                     # kl_n1 = my_kl_loss(series[u], p_stopgrad)  # KL(N || Stopgrad(P))
#                     # kl_n2 = my_kl_loss(p_stopgrad, series[u])  # KL(Stopgrad(P) || N)
#                     # prior_loss += torch.mean(kl_n1 + kl_n2)  # 双向求和
#                     # 计算 L_P = KL(P, Stopgrad(N)) + KL(Stopgrad(N), P)
#                     n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                           self.win_size)
#                     n_stopgrad = n_normalized.detach()  # 固定N的梯度
#                     kl_p1 = my_kl_loss(series[u], n_stopgrad)  # 修正：使用series[u]作为P
#                     kl_p2 = my_kl_loss(n_stopgrad, series[u])  # 修正：使用series[u]作为P
#                     series_loss += torch.mean(kl_p1 + kl_p2)  # 双向求和
#
#                     # 计算 L_N = KL(N, Stopgrad(P)) + KL(Stopgrad(P), N)
#                     p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                             self.win_size)
#                     p_stopgrad = p_normalized.detach()  # 固定P的梯度
#                     kl_n1 = my_kl_loss(prior[u], p_stopgrad)  # 修正：使用prior[u]作为N
#                     kl_n2 = my_kl_loss(p_stopgrad, prior[u])  # 修正：使用prior[u]作为N
#                     prior_loss += torch.mean(kl_n1 + kl_n2)  # 双向求和
#
#                 series_loss = series_loss / len(prior)
#                 prior_loss = prior_loss / len(prior)
#
#                 loss = prior_loss - series_loss
#
#                 if (i + 1) % 100 == 0:
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
#                     print('\tEpoch: {:d}; speed: {:.4f}s/iter; left time: {:.4f}s; Loss: {:.4f}'.format(
#                         epoch + 1,  # 当前epoch（从1开始）
#                         speed,
#                         left_time,
#                         loss.item()
#                     ))
#                     iter_count = 0
#                     time_now = time.time()
#
#                 loss.backward()
#                 self.optimizer.step()
#
#             vali_loss1, vali_loss2 = self.vali(self.vali_loader)
#
#             print(
#                 "Epoch: {0}, Cost time: {1:.3f}s ".format(
#                     epoch + 1, time.time() - epoch_time))
#             early_stopping(vali_loss1, vali_loss2, self.model, path)
#             if early_stopping.early_stop:
#                 break
#             adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
#
#     def test(self):
#         self.model.load_state_dict(
#             torch.load(
#                 os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
#         self.model.eval()
#         temperature = 50
#         attens_energy = []
#         for i, (input_data, labels) in enumerate(self.train_loader):
#             input = input_data.float().to(self.device)
#             series, prior = self.model(input)
#             series_loss = 0.0
#             prior_loss = 0.0
#             for u in range(len(prior)):
#                 n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                       self.win_size)
#                 p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                         self.win_size)
#
#                 series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
#                 prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature
#
#             metric = torch.softmax((-series_loss - prior_loss), dim=-1)
#             cri = metric.detach().cpu().numpy()
#             attens_energy.append(cri)
#
#         attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#         train_energy = np.array(attens_energy)
#
#         attens_energy = []
#         for i, (input_data, labels) in enumerate(self.thre_loader):
#             input = input_data.float().to(self.device)
#             series, prior = self.model(input)
#             series_loss = 0.0
#             prior_loss = 0.0
#             for u in range(len(prior)):
#                 n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                       self.win_size)
#                 p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                         self.win_size)
#                 series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
#                 prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature
#
#             metric = torch.softmax((-series_loss - prior_loss), dim=-1)
#             cri = metric.detach().cpu().numpy()
#             attens_energy.append(cri)
#
#         attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#         test_energy = np.array(attens_energy)
#         combined_energy = np.concatenate([train_energy, test_energy], axis=0)
#         thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
#         print("Threshold :", thresh)
#
#         test_labels = []
#         attens_energy = []
#         for i, (input_data, labels) in enumerate(self.thre_loader):
#             input = input_data.float().to(self.device)
#             series, prior = self.model(input)
#             series_loss = 0.0
#             prior_loss = 0.0
#             for u in range(len(prior)):
#                 n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                       self.win_size)
#                 p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                         self.win_size)
#                 series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
#                 prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature
#
#             metric = torch.softmax((-series_loss - prior_loss), dim=-1)
#             cri = metric.detach().cpu().numpy()
#             attens_energy.append(cri)
#             test_labels.append(labels)
#
#         attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#         test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
#         test_energy = np.array(attens_energy)
#         test_labels = np.array(test_labels)
#
#         pred = (test_energy > thresh).astype(int)
#         gt = test_labels.astype(int)
#
#         matrix = [self.index]
#         scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
#         for key, value in scores_simple.items():
#             matrix.append(value)
#             print('{0:21} : {1:0.4f}'.format(key, value))
#
#         anomaly_state = False
#         for i in range(len(gt)):
#             if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#                 anomaly_state = True
#                 for j in range(i, 0, -1):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#                 for j in range(i, len(gt)):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#             elif gt[i] == 0:
#                 anomaly_state = False
#             if anomaly_state:
#                 pred[i] = 1
#
#         pred = np.array(pred)
#         gt = np.array(gt)
#
#         accuracy = accuracy_score(gt, pred)
#         precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
#         print(
#             "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
#                                                                                                    recall, f_score))
#
#         if self.data_path == 'UCR' or 'UCR_AUG':
#             import csv
#             with open('result/' + self.data_path + '.csv', 'a+') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(matrix)
#
#         return accuracy, precision, recall, f_score
#
#     # def test(self):
#     #     self.model.load_state_dict(
#     #         torch.load(
#     #             os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
#     #     self.model.eval()
#     #     temperature = 50
#     #
#     #     # (1) stastic on the train set
#     #     attens_energy = []
#     #     for i, (input_data, labels) in enumerate(self.train_loader):
#     #         input = input_data.float().to(self.device)
#     #         series, prior = self.model(input)
#     #         series_loss = 0.0
#     #         prior_loss = 0.0
#     #         for u in range(len(prior)):
#     #             # if u == 0:
#     #             #     series_loss = my_kl_loss(series[u], (
#     #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                    self.win_size)).detach()) * temperature
#     #             #     prior_loss = my_kl_loss(
#     #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                 self.win_size)),
#     #             #         series[u].detach()) * temperature
#     #             # else:
#     #             #     series_loss += my_kl_loss(series[u], (
#     #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                    self.win_size)).detach()) * temperature
#     #             #     prior_loss += my_kl_loss(
#     #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                 self.win_size)),
#     #             #         series[u].detach()) * temperature
#     #             # 获取两个分支的表示
#     #             P = series[u]
#     #             N = prior[u]
#     #             # 计算P到N的KL散度（N停止梯度），保留原始归一化操作
#     #             N_normalized = N / torch.unsqueeze(torch.sum(N, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
#     #             kl_p_to_n = my_kl_loss(P, N_normalized.detach())  # 对应原代码中的series_loss
#     #
#     #             # 计算N到P的KL散度（P停止梯度），保留原始归一化操作
#     #             P_normalized = P / torch.unsqueeze(torch.sum(P, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
#     #             kl_n_to_p = my_kl_loss(N, P_normalized.detach())  # 对应原代码中的prior_loss
#     #
#     #             # 原代码逻辑：根据u是否为0决定初始化或累加
#     #             if u == 0:
#     #                 series_loss = kl_p_to_n
#     #                 prior_loss = kl_n_to_p
#     #             else:
#     #                 series_loss += kl_p_to_n
#     #                 prior_loss += kl_n_to_p
#     #
#     #         # 应用温度系数（移到循环外，避免重复缩放）
#     #         series_loss = series_loss * temperature
#     #         prior_loss = prior_loss * temperature
#     #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
#     #         cri = metric.detach().cpu().numpy()
#     #         attens_energy.append(cri)
#     #
#     #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#     #     train_energy = np.array(attens_energy)
#     #
#     #     # (2) find the threshold
#     #     attens_energy = []
#     #     for i, (input_data, labels) in enumerate(self.thre_loader):
#     #         input = input_data.float().to(self.device)
#     #         series, prior = self.model(input)
#     #         series_loss = 0.0
#     #         prior_loss = 0.0
#     #         for u in range(len(prior)):
#     #             # if u == 0:
#     #             #     series_loss = my_kl_loss(series[u], (
#     #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                    self.win_size)).detach()) * temperature
#     #             #     prior_loss = my_kl_loss(
#     #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                 self.win_size)),
#     #             #         series[u].detach()) * temperature
#     #             # else:
#     #             #     series_loss += my_kl_loss(series[u], (
#     #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                    self.win_size)).detach()) * temperature
#     #             #     prior_loss += my_kl_loss(
#     #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                 self.win_size)),
#     #             #         series[u].detach()) * temperature
#     #             # 获取两个分支的表示
#     #             P = series[u]
#     #             N = prior[u]
#     #             # 计算P到N的KL散度（N停止梯度），保留原始归一化操作
#     #             N_normalized = N / torch.unsqueeze(torch.sum(N, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
#     #             kl_p_to_n = my_kl_loss(P, N_normalized.detach())  # 对应原代码中的series_loss
#     #
#     #             # 计算N到P的KL散度（P停止梯度），保留原始归一化操作
#     #             P_normalized = P / torch.unsqueeze(torch.sum(P, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
#     #             kl_n_to_p = my_kl_loss(N, P_normalized.detach())  # 对应原代码中的prior_loss
#     #
#     #             # 原代码逻辑：根据u是否为0决定初始化或累加
#     #             if u == 0:
#     #                 series_loss = kl_p_to_n
#     #                 prior_loss = kl_n_to_p
#     #             else:
#     #                 series_loss += kl_p_to_n
#     #                 prior_loss += kl_n_to_p
#     #
#     #                 # 应用温度系数（移到循环外，避免重复缩放）
#     #         series_loss = series_loss * temperature
#     #         prior_loss = prior_loss * temperature
#     #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
#     #         cri = metric.detach().cpu().numpy()
#     #         attens_energy.append(cri)
#     #
#     #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#     #     test_energy = np.array(attens_energy)
#     #     combined_energy = np.concatenate([train_energy, test_energy], axis=0)
#     #     thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
#     #     print("Threshold :", thresh)
#     #
#     #     # (3) evaluation on the test set
#     #     test_labels = []
#     #     attens_energy = []
#     #     for i, (input_data, labels) in enumerate(self.thre_loader):
#     #         input = input_data.float().to(self.device)
#     #         series, prior = self.model(input)
#     #         series_loss = 0.0
#     #         prior_loss = 0.0
#     #         for u in range(len(prior)):
#     #             # if u == 0:
#     #             #     series_loss = my_kl_loss(series[u], (
#     #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                    self.win_size)).detach()) * temperature
#     #             #     prior_loss = my_kl_loss(
#     #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                 self.win_size)),
#     #             #         series[u].detach()) * temperature
#     #             # else:
#     #             #     series_loss += my_kl_loss(series[u], (
#     #             #             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                    self.win_size)).detach()) * temperature
#     #             #     prior_loss += my_kl_loss(
#     #             #         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#     #             #                                                                                 self.win_size)),
#     #             #         series[u].detach()) * temperature
#     #             # 获取两个分支的表示
#     #             P = series[u]
#     #             N = prior[u]
#     #             # 计算P到N的KL散度（N停止梯度），保留原始归一化操作
#     #             N_normalized = N / torch.unsqueeze(torch.sum(N, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
#     #             kl_p_to_n = my_kl_loss(P, N_normalized.detach())  # 对应原代码中的series_loss
#     #
#     #             # 计算N到P的KL散度（P停止梯度），保留原始归一化操作
#     #             P_normalized = P / torch.unsqueeze(torch.sum(P, dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
#     #             kl_n_to_p = my_kl_loss(N, P_normalized.detach())  # 对应原代码中的prior_loss
#     #
#     #             # 原代码逻辑：根据u是否为0决定初始化或累加
#     #             # 原代码逻辑：根据u是否为0决定初始化或累加
#     #             if u == 0:
#     #                 series_loss = kl_p_to_n
#     #                 prior_loss = kl_n_to_p
#     #             else:
#     #                 series_loss += kl_p_to_n
#     #                 prior_loss += kl_n_to_p
#     #
#     #                 # 应用温度系数（移到循环外，避免重复缩放）
#     #         series_loss = series_loss * temperature
#     #         prior_loss = prior_loss * temperature
#     #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
#     #         cri = metric.detach().cpu().numpy()
#     #         attens_energy.append(cri)
#     #         test_labels.append(labels)
#     #
#     #     attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#     #     test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
#     #     test_energy = np.array(attens_energy)
#     #     test_labels = np.array(test_labels)
#     #
#     #     pred = (test_energy > thresh).astype(int)
#     #     gt = test_labels.astype(int)
#     #
#     #     matrix = [self.index]
#     #     scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
#     #     for key, value in scores_simple.items():
#     #         matrix.append(value)
#     #         print('{0:21} : {1:0.4f}'.format(key, value))
#     #
#     #     anomaly_state = False
#     #     for i in range(len(gt)):
#     #         if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#     #             anomaly_state = True
#     #             for j in range(i, 0, -1):
#     #                 if gt[j] == 0:
#     #                     break
#     #                 else:
#     #                     if pred[j] == 0:
#     #                         pred[j] = 1
#     #             for j in range(i, len(gt)):
#     #                 if gt[j] == 0:
#     #                     break
#     #                 else:
#     #                     if pred[j] == 0:
#     #                         pred[j] = 1
#     #         elif gt[i] == 0:
#     #             anomaly_state = False
#     #         if anomaly_state:
#     #             pred[i] = 1
#     #
#     #     pred = np.array(pred)
#     #     gt = np.array(gt)
#     #
#     #     from sklearn.metrics import precision_recall_fscore_support
#     #     from sklearn.metrics import accuracy_score
#     #
#     #     accuracy = accuracy_score(gt, pred)
#     #     precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
#     #     print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
#     #
#     #     if self.data_path == 'UCR' or 'UCR_AUG':
#     #         import csv
#     #         with open('result/'+self.data_path+'.csv', 'a+') as f:
#     #             writer = csv.writer(f)
#     #             writer.writerow(matrix)
#     #
#     #     return accuracy, precision, recall, f_score