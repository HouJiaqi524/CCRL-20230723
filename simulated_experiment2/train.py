# coding=utf-8
# @FileName  :train_model.py
# @Time      :2022/8/5 8:58
# @Author    :

import scipy.stats
import torch
from torch import optim
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import ToyCCRL
from utils import setup_seed, get_loader, plot_original, train, test

# random seed
setup_seed(7)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### datasets import,  circles datasets
X, y = make_circles(n_samples=10000, factor=0.5, noise=0.05)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, stratify=y)

# model params
input_dim1 = 2
input_dim2 = 1
output_dim1 = output_dim2 = 2
batch_size = 500
lr = 0.001
config = {
    'a1': 4,
    'a2': 1 / 10,
}  # hyperparams of loss

# model
ccrl = ToyCCRL(input_dim1, input_dim2, output_dim1, output_dim2)
ccrl = ccrl.to(device)

# optimizer
ccrl_optim = optim.Adam(ccrl.parameters(), lr=lr, weight_decay=1e-4)

# dataloader
train_loader, test_loader = get_loader(train_x, train_y, test_x, test_y, batch_size)

plt.figure(figsize=(10, 10))
lists = [0, 9, 49, 99]
# lists = [0, 2, 3, 4]
count = 0

for epoch in range(100):
    train_ksi, train_eta, loss = train(ccrl, ccrl_optim, config, train_loader, device)
    test_ksi, test_eta, r = test(ccrl, test_loader, device)
    print(f'epoch={epoch}, train_loss={loss}, test_spearmanr={r}')
    if epoch in lists:
        count += 1
        plt.subplot(4, 2, 2*count-1)
        if epoch==0:
            plot_original(train_ksi, train_y, test_x, test_y)
        else:
            plot_original(train_ksi, train_y, test_ksi, test_y)

        plt.title(f"Epoch={epoch}({round(r, 2)})", y=-0.3)
# plt.savefig('fig/1.eps')

setup_seed(7)
del train_x, test_x, train_y, test_y


######## data import  two-moon datasets
X1, y1 = make_moons(n_samples=10000, noise=0.05)
train_x, test_x, train_y, test_y = train_test_split(X1, y1, test_size=0.2, stratify=y)

# model params
input_dim1 = 2
input_dim2 = 1
output_dim1 = output_dim2 = 2
batch_size = 500
lr = 0.001
config = {
    'a1': 4,
    'a2': 1 / 5,
}  # hyperparams of loss

# model
ccrl1 = ToyCCRL(input_dim1, input_dim2, output_dim1, output_dim2)
ccrl1 = ccrl1.to(device)

# optimizer
ccrl_optim1 = optim.Adam(ccrl1.parameters(), lr=lr, weight_decay=1e-4)

# dataloader
train_loader, test_loader = get_loader(train_x, train_y, test_x, test_y, batch_size)

lists = [0, 9, 49, 119]
count = 0
for epoch in range(120):
    train_ksi, train_eta, loss = train(ccrl1, ccrl_optim1, config, train_loader, device)
    test_ksi, test_eta, r = test(ccrl1, test_loader, device)
    print(f'epoch={epoch}, train_loss={loss}, test_spearmanr={r}')
    if epoch in lists:
        count += 1
        plt.subplot(4, 2, 2*count)
        if epoch==0:
            plot_original(train_ksi, train_y, test_x, test_y)
        else:
            plot_original(train_ksi, train_y, test_ksi, test_y)
        plt.title(f"Epoch={epoch}({round(r, 2)})", y=-0.3)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('fig/figure3.eps')



