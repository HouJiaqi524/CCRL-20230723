import torch
import torch.nn as nn


def CCRL_loss(config, u1, u2, out=None, batch_y=None):
    loss_duli = 0
    loss_uniform = 0
    loss_co = 0
    for j in range(u1.shape[1]):
        A = torch.unsqueeze(u1[:, j], 1)
        B = torch.unsqueeze(u2[:, j], 1)

        # independence constraint of each two components of U and V.
        for k in range(j + 1, u1.shape[1]):
            C = torch.unsqueeze(u1[:, k], 1)
            D = torch.unsqueeze(u2[:, k], 1)
            # print(12*torch.mean(torch.mul(A, C))-3)
            loss_duli = loss_duli + torch.abs(12 * torch.mean(torch.mul(A, C)) - 3) \
                                  + torch.abs(12 * torch.mean(torch.mul(B, D)) - 3)

        # U（0，1）constraints
        A_mean = torch.mean(A)
        A_std = torch.std(A)
        B_mean = torch.mean(B)
        B_std = torch.std(B)
        loss_uniform = loss_uniform + ((A_mean - 0.5) ** 2 + (B_mean - 0.5) ** 2) + (
                (A_std - 1 / 12 ** 0.5) ** 2 + (B_std - 1 / 12 ** 0.5) ** 2)

        # L2 distance between the distribution of (ui,vi) and co-monotonicity copula
        loss_co = loss_co + torch.mean(torch.mul(A - B, A - B))

    if out is not None:
        # loss_td
        loss = nn.CrossEntropyLoss()
        loss_TD = loss(out, batch_y.long())

        # total loss
        generator_loss = config["a1"] * loss_co + config["a2"] * loss_uniform + config["a3"] * loss_duli + loss_TD

        return config["a1"] * loss_co, config["a2"] * loss_uniform, config["a3"] * loss_duli, loss_TD, generator_loss
    else:
        # total loss
        generator_loss = config["a1"] * loss_co + config["a2"] * loss_uniform + config["a3"] * loss_duli

        return config["a1"] * loss_co, config["a2"] * loss_uniform, config["a3"] * loss_duli, generator_loss
