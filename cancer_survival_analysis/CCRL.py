# coding=utf-8
# @FileName  :CCRL.py
# @Time      :2022/8/2 12:41
#

import numpy as np
import torch
import torch.nn as nn
from MonotonicNN import MonotonicNN


class Generator(nn.Module):
    def __init__(self, input_dim: int = 273, output_dim: int = 112, dataset: str = 'XRMB'):
        super(Generator, self).__init__()
        self.dataset = dataset
        self.dense1 = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, output_dim)
        )
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, output_dim)
        )

    def forward(self, x):
        if self.dataset == 'XRMB':
            ksi = self.dense(x)
        else:
            ksi = self.dense1(x)
        return ksi


class Mapping(nn.Module):
    def __init__(self, input_dim: int = 273, output_dim: int = 112, dataset: str = 'XRMB', device=torch.device('cpu')):
        super(Mapping, self).__init__()
        self.output_dim = output_dim
        self.gen = Generator(input_dim, output_dim, dataset).to(device)
        self.device = device

        keys = np.arange(1, self.output_dim + 1)  # [ 1  2  3 ... 49 50]
        keys = [str(i) for i in keys]
        keys = tuple(keys)
        self.dic = dict.fromkeys(keys, MonotonicNN(3, [50, 50], nb_steps=100, dev=self.device))
        for attr, value in self.dic.items():
            setattr(self, attr, value)

    def forward(self, x):
        count = 0
        ksi = self.gen(x)
        for attr, _ in self.dic.items():
            u_i = getattr(self, attr)(torch.unsqueeze(ksi[:, count], 1), torch.ones(ksi.shape[0], 2).to(self.device))
            u_i = (u_i - torch.min(u_i)) / (torch.max(u_i) - torch.min(u_i))  # min_max_scaling

            if count == 0:
                u = u_i
            else:
                u = torch.cat((u, u_i), dim=1)
            count = count + 1

        return ksi, u


class CCRL(nn.Module):
    def __init__(self, input_dim1: int = 273, input_dim2: int = 112, dataset: str = 'XRMB',
                 output_dim1: int = 112, output_dim2: int = 112, task=None, class_num=40, device=torch.device('cpu')):  # TODO:
        super(CCRL, self).__init__()
        self.device = device
        self.model1 = Mapping(input_dim1, output_dim1, dataset, self.device)
        self.model2 = Mapping(input_dim2, output_dim2, dataset, self.device)
        self.task = task
        if self.task is None:
            pass
        elif (self.task == 'L2R') | (self.task == 'R2L'):
            self.linear1 = nn.Linear(output_dim1, class_num)
        elif self.task == 'SupervisedDataFusion':
            self.linear1 = nn.Linear(output_dim1+output_dim2, class_num)
        else:
            exec('print("incompatible task")')

    def forward(self, x1, x2):
        ksi, u1 = self.model1(x1)
        eta, u2 = self.model2(x2)
        if self.task is None:
            pass
        elif self.task == 'L2R':
            out = self.linear1(ksi)
        elif self.task == 'R2L':
            out = self.linear1(eta)
        elif self.task == 'SupervisedDataFusion':
            fea = torch.concat((ksi, eta), dim=1)
            out = self.linear1(fea)
        else:
            exec('incompatible task')

        try:
            return ksi, eta, u1, u2, out
        except:
            return ksi, eta, u1, u2