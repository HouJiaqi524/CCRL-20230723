import numpy as np
import torch
import torch.nn as nn
from MonotonicNN import MonotonicNN


class XNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(XNet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.BatchNorm1d(6),

            nn.Linear(6, 6),
            nn.ReLU(),
            nn.BatchNorm1d(6),

            nn.Linear(6, 6),
            nn.ReLU(),
            nn.BatchNorm1d(6),

            nn.Linear(6, output_dim)
        )

    def forward(self, x):
        return self.dense(x)


class YNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(YNet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, y):
        out = self.dense(y)
        return out


class Mapping(nn.Module):
    def __init__(self, info: 'y'or'x', input_dim: int, output_dim: int, device=torch.device('cpu')):
        super(Mapping, self).__init__()
        self.output_dim = output_dim
        if info == 'x':
            self.net = XNet(input_dim, output_dim)
        else:
            self.net = YNet(input_dim, output_dim)

        self.device = device

        keys = np.arange(1, self.output_dim + 1)  # [ 1  2  3 ... 49 50]
        keys = [str(i) for i in keys]
        keys = tuple(keys)
        self.dic = dict.fromkeys(keys, MonotonicNN(3, [50, 50], nb_steps=100, dev=self.device))
        for attr, value in self.dic.items():
            setattr(self, attr, value)

    def forward(self, x):
        count = 0
        ksi = self.net(x)
        for attr, _ in self.dic.items():
            u_i = getattr(self, attr)(torch.unsqueeze(ksi[:, count], 1), torch.ones(ksi.shape[0], 2).to(self.device))
            u_i = (u_i - torch.min(u_i)) / (torch.max(u_i) - torch.min(u_i))  # min_max_scaling

            if count == 0:
                u = u_i
            else:
                u = torch.cat((u, u_i), dim=1)
            count = count + 1

        return ksi, u


class ToyCCRL(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int,
                 output_dim1: int, output_dim2: int, device=torch.device('cpu')):
        super(ToyCCRL, self).__init__()
        self.device = device
        self.model1 = Mapping('x', input_dim1, output_dim1, self.device)
        self.model2 = Mapping('y', input_dim2, output_dim2, self.device)

    def forward(self, x1, x2):
        ksi, u1 = self.model1(x1)
        eta, u2 = self.model2(x2)
        return ksi, eta, u1, u2

