import torch
import numpy as np
import random
import torch.utils.data as Data
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr

from objective import CCRL_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_loader(train_x, train_y, test_x, test_y, batch_size):
    train_x = torch.tensor(train_x, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.float).unsqueeze(1)
    test_x = torch.tensor(test_x, dtype=torch.float)
    test_y = torch.tensor(test_y, dtype=torch.float).unsqueeze(1)
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_x, train_y),
                                   batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_x, test_y),
                                  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def plot_original(train_x, train_y, test_x, test_y):
    # plt.figure()
    # _, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    #
    # train_ax.scatter(train_x[:, 0], train_x[:, 1], c=train_y)
    # train_ax.set_ylabel("Feature #1")
    # train_ax.set_xlabel("Feature #0")
    # train_ax.set_title("Training data")
    #
    # test_ax.scatter(test_x[:, 0], test_x[:, 1], c=test_y)
    # test_ax.set_xlabel("Feature #0")
    # _ = test_ax.set_title("Testing data")
    # plt.show()
    col = []
    colors = ['#2A419E', '#F79011']
    for i in range(0, len(test_y)):
        col.append(colors[test_y[i]])

    map_size = {0: 10, 1: 10}
    size = list(map(lambda x: map_size[x], test_y))
    map_color = {0: '#2A419E', 1: '#F79011'}
    color = list(map(lambda x: map_color[x], test_y))
    map_marker = {0: 'o', 1: '^'}
    markers = list(map(lambda x: map_marker[x], test_y))
    mscatter(np.array(test_x[:, 0]), np.array(test_x[:, 1]), s=size, c=color, m=markers)

    # plt.scatter(test_x[:, 0], test_x[:, 1], c=col, )
    # plt.ylabel("Feature #1")
    # plt.xlabel("Feature #0")






def train(model, optimizer, config, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()

    for batch, (batch_x1, batch_x2) in tqdm(enumerate(train_loader)):
        # to gpu
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)

        optimizer.zero_grad()
        #
        ksi, eta, u1, u2 = model(batch_x1, batch_x2)
        #
        # loss_co, loss_uniform, loss_duli, loss_TD, generator_loss = TD_CCRL_loss(u1, u2, out1, batch_y)
        loss_co, loss_uniform, loss_duli, loss = CCRL_loss(config, u1, u2)

        print('loss_co:', loss_co)
        print('loss_uniform:', loss_uniform)
        print('loss_duli:',loss_duli)
        if batch == 0:
            Ksi, Eta = ksi, eta
        else:
            Ksi = torch.cat((Ksi, ksi), dim=0)
            Eta = torch.cat((Eta, eta), dim=0)
        loss.backward()
        optimizer.step()
    Ksi, Eta = Ksi.detach().numpy(), Eta.detach().numpy()
    return Ksi, Eta, loss


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    with torch.no_grad():
        for i, (batch_x1, batch_x2) in tqdm(enumerate(data_loader)):
            # to gpu
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            #
            ksi, eta, _, _ = model(batch_x1, batch_x2)
            #
            if i == 0:
                Ksi, Eta = ksi, eta
            else:
                Ksi = torch.cat((Ksi, ksi), dim=0)
                Eta = torch.cat((Eta, eta), dim=0)
        Ksi, Eta = Ksi.detach().numpy(), Eta.detach().numpy()

    r = 0
    for i in range(Ksi.shape[1]):
        r = r + spearmanr(Ksi[:, i], Eta[:, i])[0]

    return Ksi, Eta, r
