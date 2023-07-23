# coding=utf-8
# @FileName  :train_model.py
# @Time      :2022/8/5 8:58

import os
from functools import partial

import torch
from ray import tune
from scipy.stats import spearmanr
from torch import optim
import torch.nn as nn
from tqdm import tqdm

from objective import CCRL_loss
from Dataloader import create_datasets
from CCRL import CCRL


def train_xrmb(config, checkpoint_dir=None):
    # Data Setup

    #################TODO: specific for two datasets ####################
    # # MNIST pre params for create_datasets()
    batch_size = 500
    shuffle = True
    string = 'MNIST'

    # # XRMB pre params for create_datasets()
    # batch_size = 1000
    # shuffle = True
    # string = 'XRMB'

    train_loader, _, valid_loader = create_datasets(batch_size, shuffle, string=string)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # XRMB params for model train
    # dataset = 'XRMB'
    # input_dim1 = 273
    # input_dim2 = 112
    # output_dim1 = output_dim2 = 112
    # hidden_dim = 500
    # lr = 0.001
    # wd = 1e-4
    # task = 'SupervisedDataFusion'
    # class_num = 40

    # MNIST params for model train
    dataset = 'MNIST'
    input_dim1 = 392
    input_dim2 = 392
    output_dim1 = output_dim2 = 50
    hidden_dim = 500
    lr = 0.001
    wd = 0
    task = None # None ,or 'L2R',or 'R2L'
    class_num = 10
    #################TODO: specific for two datasets ####################

    # model
    ccrl = CCRL(input_dim1, input_dim2, dataset, output_dim1, output_dim2, task=task, class_num=class_num, device=device)
    ccrl = ccrl.to(device)

    # optimizer
    ccrl_optim = optim.Adam(ccrl.parameters(), lr=lr, weight_decay=wd)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        ccrl.load_state_dict(model_state)
        ccrl_optim.load_state_dict(optimizer_state)

    opt_acc = float("-inf")
    opt_cor = float("-inf")
    opt_loss = float("+inf")
    for epoch in range(500):

        loss = train(ccrl, ccrl_optim, config, train_loader, task, device)
        if task is None:
            val_loss, cor = test(ccrl,config, valid_loader, task, device)
            print(f'epoch={epoch}, total corr in validation set ={cor}')
        else:
            val_loss, cor, acc = test(ccrl,config, valid_loader, task, device)
            print(f'epoch={epoch}, acc in validation set ={acc}, total corr in validation set ={cor}')

        if val_loss < opt_loss:
            if string == 'MNIST':
                path = rf'out/MNIST/{task}opt_loss.pt'
            elif string == 'XRMB':
                path = rf'out/XRMB/{task}opt_loss.pt'
            else:
                path = rf'out/others/{task}opt_loss.pt'
            print(f'Validation loss decreased ({opt_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            state = {'model': ccrl.state_dict(), 'optimizer': ccrl_optim.state_dict(), 'epoch': epoch,
                     'train_loss': loss, 'valid_loss': val_loss,
                     'valid_cor': cor}
            torch.save(state, path)
            # opt_cor = cor
            opt_loss = val_loss
            count = 0
        else:
            count += 1

        # 早停：
        if count >= 20:
            break;

        # if cor > opt_cor:
        #     if string == 'MNIST':
        #         path = rf'out/MNIST/{task}opt_cor.pt'
        #     elif string == 'XRMB':
        #         path = rf'out/XRMB/{task}opt_cor.pt'
        #     else:
        #         path = rf'out/others/{task}opt_cor.pt'
        #     print(f'Validation cor decreased ({opt_cor:.6f} --> {cor:.6f}).  Saving model ...')
        #     state = {'model': ccrl.state_dict(), 'optimizer': ccrl_optim.state_dict(), 'epoch': epoch,
        #              'train_loss': loss, 'valid_cor': cor}
        #     torch.save(state, path)
        #     opt_cor = cor
        try:
            if acc > opt_acc:
                if string == 'MNIST':
                    path = rf'out/MNIST/{task}opt_acc.pt'
                elif string == 'XRMB':
                    path = rf'out/XRMB/{task}opt_acc.pt'
                else:
                    path = rf'out/others/{task}opt_acc.pt'
                print(f'Validation acc decreased ({opt_acc:.6f} --> {acc:.6f}).  Saving model ...')
                state = {'model': ccrl.state_dict(), 'optimizer': ccrl_optim.state_dict(), 'epoch': epoch,
                         'train_loss': loss, 'valid_acc': acc}
                torch.save(state, path)
                opt_acc = acc
        except:
            pass

        # with tune.checkpoint_dir(i) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((ccrl.state_dict(), ccrl_optim.state_dict()), path)

        # # Send the current training result back to Tune
        # tune.report(mean_accuracy=acc)

def train(model, optimizer, config, train_loader, task=None, device=None):
    device = device or torch.device("cpu")
    model.train()

    for batch, (batch_x1, batch_x2, batch_y) in tqdm(enumerate(train_loader)):
        # to GPU
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_y = batch_y.to(device)
        #
        optimizer.zero_grad()
        # model training
        if task is None:
            _, _, u1, u2 = model(batch_x1, batch_x2)
            loss_co, loss_uniform, loss_duli, loss = CCRL_loss(config=config, u1=u1, u2=u2)
            print('loss_co:', loss_co)
            print('loss_uniform:', loss_uniform)
            print('loss_duli:', loss_duli)
            print('loss', loss)
        else:
            _, _, u1, u2, out = model(batch_x1, batch_x2)
            loss_co, loss_uniform, loss_duli, loss_TD, loss = CCRL_loss(config=config, u1=u1, u2=u2,
                                                                        out=out, batch_y=batch_y)
            print('loss_co:', loss_co)
            print('loss_uniform:', loss_uniform)
            print('loss_duli:',loss_duli)
            print('loss_TD:',loss_TD)
            print('loss', loss)

        loss.backward()
        optimizer.step()
    return loss


def test(model,config, data_loader,task=None, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    total_sum = 0
    with torch.no_grad():
        for batch, (batch_x1, batch_x2, batch_y) in tqdm(enumerate(data_loader)):
            # to GPU
            batch_x1 = batch_x1.to(device)
            batch_x2 = batch_x2.to(device)
            batch_y = batch_y.to(device)
            # model training
            if task is None:
                _, _, u1, u2 = model(batch_x1, batch_x2)
                loss_co, loss_uniform, loss_duli, loss = CCRL_loss(config=config, u1=u1, u2=u2)
                for i in range(u1.shape[1]):
                    total_sum = total_sum + spearmanr(u1[:, i], u2[:, i])[0]

            else:
                _, _, u1, u2, out = model(batch_x1, batch_x2)
                # loss in valid set
                _, predicted1 = torch.max(out.data, 1)
                total += batch_y.size(0)
                correct += (predicted1 == batch_y).sum().item()
                for i in range(u1.shape[1]):
                    total_sum = total_sum + spearmanr(u1[:, i].cpu().detach().numpy(), u2[:, i].cpu().detach().numpy())[0]
        if task is None:
            return loss, total_sum/batch
        else:
            return loss, total_sum/batch, correct / total
