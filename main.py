import os
import argparse
import random

import numpy as np
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import time
import warnings

warnings.filterwarnings('ignore')

import sys


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def str2bool(v):
    return v.lower() in ('true')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx - 1])


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        config.istrain = True
        solver.train()
    if config.istest:
        solver.mode = 'test'
        solver.istrain = False
        solver.test()

    return solver


def set_seed(seed: int):
    """固定各种随机种子，使结果稳定可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况

    # 下面两项设置可以保证PyTorch使用确定性算法，避免非确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Alternative
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--patch_size', type=list, default=[5])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--rec_timeseries', action='store_true', default=True)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Default
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_c', type=int, default=9)
    parser.add_argument('--output_c', type=int, default=9)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--istest', action='store_true', default=True,
                        help='Enable testing mode (default: True). Use --no-istest to disable.')
    parser.add_argument('--istrain', action='store_true', default=True,
                        help='Enable testing mode (default: True). Use --no-istest to disable.')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--contrastive_weight', type=float, default=0.15)
    parser.add_argument('--sigma', type=float, default=5.0)
    config = parser.parse_args()
    args = vars(config)
    config.patch_size = [int(patch_index) for patch_index in config.patch_size]

    if config.dataset == 'UCR':
        batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256]
        data_len = np.load('dataset/' + config.data_path + "/UCR_" + str(config.index) + "_train.npy").shape[0]
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    elif config.dataset == 'UCR_AUG':
        batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256]
        data_len = np.load('dataset/' + config.data_path + "/UCR_AUG_" + str(config.index) + "_train.npy").shape[0]
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    elif config.dataset == 'SMD_Ori':
        batch_size_buffer = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        data_len = np.load('dataset/' + config.data_path + "/SMD_Ori_" + str(config.index) + "_train.npy").shape[0]
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)

    # config.dataset = 'EV47'
    if config.dataset == 'SWAT':
        config.anormly_ratio = 1
        config.num_epochs = 1
        config.batch_size = 64  # 根据内存来该改
        config.mode = 'train'
        config.data_path = 'SWAT'
        config.input_c = 51
        config.output_c = 51
        config.loss_fuc = 'MSE'
        config.patch_size = [1,3]
        config.win_size = 54

    if config.dataset == 'EV47':
        config.anormly_ratio = 1
        config.num_epochs =1
        config.batch_size = 64# 根据内存来该改
        config.mode = 'test'
        config.data_path = config.dataset
        config.input_c = 22
        config.output_c = 22
        config.e_layers = 3
        config.n_heads = 1
        config.d_model = 256
        config.loss_fuc = 'MSE'
        config.patch_size = [1,7]
        config.win_size = 70 # 是3,5,7的最小公倍数
        config.contrastive_weight = 0.05
        config.dropout = 0.0
        config.sigma = 5.0

        # set_seed(18)

    if config.dataset == 'EV24':
        config.anormly_ratio = 1
        config.num_epochs = 1
        config.batch_size = 64  # 根据内存来该改
        config.mode = 'train'
        config.data_path = config.dataset
        config.input_c = 22
        config.output_c = 22
        config.e_layers = 3
        config.n_heads = 1
        config.d_model = 256
        config.loss_fuc = 'MSE'
        config.patch_size = [1,7]#[5,7]
        config.win_size = 70# 是3,5,7的最小公倍数[70]
        config.istest = True
        config.contrastive_weight = 0.03#0.03
        config.dropout = 0.01
        config.lr = 1e-4

    if config.dataset == 'VLoong':
        config.anormly_ratio = 1
        config.num_epochs = 1
        config.batch_size = 64  # 根据内存来该改
        config.mode = 'test'
        config.data_path = config.dataset
        config.input_c = 8
        config.output_c = 8
        config.loss_fuc = 'MSE'
        config.patch_size = [3,5,7]
        config.win_size = 105
        config.contrastive_weight = 0.15

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    sys.stdout = Logger("./result/" + config.data_path + ".log", sys.stdout)
    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('====================  Train  ===================')

    main(config)


