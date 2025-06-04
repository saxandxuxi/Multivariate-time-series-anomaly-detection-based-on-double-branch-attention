import torch
import torch.nn as nn
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 请用户自行实现以下辅助函数
# from your_module import get_loader_segment, my_kl_loss, EarlyStopping, adjust_learning_rate, combine_all_evaluation_scores

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(
            self.index,
            'dataset/' + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='train',
            dataset=self.dataset
        )
        self.vali_loader = get_loader_segment(
            self.index,
            'dataset/' + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='val',
            dataset=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.index,
            'dataset/' + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='test',
            dataset=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.index,
            'dataset/' + self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            mode='thre',
            dataset=self.dataset
        )

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = DCdetector(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            n_heads=self.n_heads,
            d_model=self.d_model,
            e_layers=self.e_layers,
            patch_size=self.patch_size,
            channel=self.input_c,
            mode=self.mode
        )

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)

            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                n_stopgrad = n_normalized.detach()
                kl_p1 = my_kl_loss(prior[u], n_stopgrad)
                kl_p2 = my_kl_loss(n_stopgrad, prior[u])
                series_loss += torch.mean(kl_p1 + kl_p2)

                p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                p_stopgrad = p_normalized.detach()
                kl_n1 = my_kl_loss(series[u], p_stopgrad)
                kl_n2 = my_kl_loss(p_stopgrad, series[u])
                prior_loss += torch.mean(kl_n1 + kl_n2)

            series_loss /= len(prior)
            prior_loss /= len(prior)
            loss = (prior_loss - series_loss)
            loss_1.append(loss.item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                    n_stopgrad = n_normalized.detach()
                    kl_p1 = my_kl_loss(prior[u], n_stopgrad)
                    kl_p2 = my_kl_loss(n_stopgrad, prior[u])
                    series_loss += torch.mean(kl_p1 + kl_p2)

                    p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                    p_stopgrad = p_normalized.detach()
                    kl_n1 = my_kl_loss(series[u], p_stopgrad)
                    kl_n2 = my_kl_loss(p_stopgrad, series[u])
                    prior_loss += torch.mean(kl_n1 + kl_n2)

                series_loss /= len(prior)
                prior_loss /= len(prior)
                loss = (prior_loss - series_loss)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tEpoch: {:d}; speed: {:.4f}s/iter; left time: {:.4f}s; Loss: {:.4f}'.format(
                        epoch + 1,
                        speed,
                        left_time,
                        loss.item()
                    ))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior, progression_matrices = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)

                series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
                prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior, progression_matrices = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
                prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior, progression_matrices = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                n_normalized = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                p_normalized = series[u] / torch.unsqueeze(torch.sum(series[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
                series_loss += my_kl_loss(series[u], n_normalized.detach()) * temperature
                prior_loss += my_kl_loss(prior[u], p_normalized.detach()) * temperature

            corr_loss = 0.0
            for prog in progression_matrices:
                corr_loss += torch.mean(torch.norm(prog, p=1, dim=(1, 2)))

            combined_score = -series_loss - prior_loss + 0.1 * corr_loss
            metric = torch.softmax(combined_score, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

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
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))

        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/' + self.data_path + '.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score

# 请用户自行实现以下类和函数
# class DCdetector(nn.Module):
#     def __init__(self, ...):
#         super().__init__()
#         # 模型结构定义
#
#     def forward(self, x):
#         # 前向传播逻辑
#         return series, prior, progression_matrices

# def get_loader_segment(index, path, batch_size, win_size, mode, dataset):
#     # 数据加载器实现
#     pass

# def my_kl_loss(p, q):
#     # KL散度计算实现
#     pass

# class EarlyStopping:
#     # 早停机制实现
#     pass

# def adjust_learning_rate(optimizer, epoch, lr):
#     # 学习率调整实现
#     pass

# def combine_all_evaluation_scores(pred, gt, test_energy):
#     # 评估指标计算实现
#     pass

# 示例使用代码
# if __name__ == "__main__":
#     config = {
#         "index": 0,
#         "data_path": "your_dataset",
#         "batch_size": 64,
#         "win_size": 100,
#         "input_c": 22,
#         "output_c": 22,
#         "n_heads": 8,
#         "d_model": 512,
#         "e_layers": 3,
#         "patch_size": 16,
#         "mode": "train",
#         "model_save_path": "checkpoints",
#         "loss_fuc": "MSE",
#         "lr": 0.0001,
#         "num_epochs": 100,
#         "anormly_ratio": 0.01,
#         "dataset": "your_dataset_type"
#     }
#
#     solver = Solver(config)
#     solver.train()
#     solver.test()