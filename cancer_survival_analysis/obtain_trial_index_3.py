# coding=utf-8
# @Time    : 2023/3/6 18:34
# @Author  : Hou Jiaqi
# @File    : obtain_trial_index.py
# @Project : CCRL_cancer_survival_clustering
import gc
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch import optim
import torch.nn as nn

import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from CCRL import CCRL
from objective import CCRL_loss


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)



def train(model, optimizer, config, x1, x2, device):
    x1 = x1.to(device)
    x2 = x2.to(device)
    model.train()

    total_corr = 0
    optimizer.zero_grad()
    _, _, u1, u2 = model(x1, x2)
    loss_co, loss_uniform, loss_duli, loss = CCRL_loss(config=config, u1=u1, u2=u2)
    for i in range(u1.shape[1]):
        try:
            total_corr = total_corr + abs(spearmanr(u1[:, i].detach().numpy(),
                                                    u2[:, i].detach().numpy())[0])
        except:
            total_corr = total_corr + abs(spearmanr(u1[:, i].cpu().detach().numpy(),
                                                    u2[:, i].cpu().detach().numpy())[0])

    print('loss_co:', loss_co)
    print('loss_uniform:', loss_uniform)
    print('loss_duli:', loss_duli)
    print('loss', loss)
    loss.backward()
    optimizer.step()
    return loss, total_corr


def test(model, config, x1, x2, device):
    x1 = x1.to(device)
    x2 = x2.to(device)
    model.eval()

    total_corr = 0
    with torch.no_grad():
        _, _, u1, u2 = model(x1, x2)
        loss_co, loss_uniform, loss_duli, loss = CCRL_loss(config=config, u1=u1, u2=u2)
        for i in range(u1.shape[1]):
            total_corr = total_corr + abs(spearmanr(u1[:, i].cpu(), u2[:, i].cpu())[0])

    return loss, total_corr


class ObtainTrialIndex():
    '''
    1. cross_validation() to make the most of the dataset where the number of the samples is small
    2. train_ccrl() for training the ccrl feature extractor
    3. successive_task() used the transformed omics feature by ccrl to conduce survival type clustering, and some
        survival analysis, the detailed implementation is by class SuccessiveTask()
    '''

    def __init__(self, config_dict, out_address: str, trial_num: int):
        self.config_dict = config_dict
        self.out_address = out_address
        self.index_df = pd.DataFrame(columns=['Fold', 'P-value', 'C-index'])
        self.trial_num = trial_num
        self.p_value = None
        self.C_index = None

    def cross_validation(self):
        # number of fold
        k = self.config_dict['fold_num']
        # data
        X1 = self.config_dict['concat_omics'].copy()
        X2 = self.config_dict['survival'].copy()
        # ##TODO: attention
        X2['time'] = (X2['time'] - X2['time'].min()) / (X2['time'].max()-X2['time'].min())
        ensemble = 31
        # survival_label = None

        k_fold = KFold(n_splits=k, shuffle=True, random_state=7)
        for k, (train_index, test_index) in enumerate(k_fold.split(X1, X2)):
            survival_label = None
            # 集成模型，使用训练集中的80%
            for ensem in range(ensemble):
                c = train_index.copy()
                np.random.shuffle(c)
                if ensem <= 20:
                    pass
                else:
                    # intermediate results to save
                    process_df = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss',
                                                       'train_cor', 'valid_cor'])
                    # a small portion of the training data for model validation
                    a = int(len(c)*0.2)
                    train_index_ = c[a:]
                    valid_index_ = c[0:a]

                    # data to torch.tensor
                    train_x1 = torch.from_numpy(X1.iloc[train_index_].values).float()
                    train_x2 = torch.from_numpy(X2.iloc[train_index_].values).float()
                    valid_x1 = torch.from_numpy(X1.iloc[valid_index_].values).float()
                    valid_x2 = torch.from_numpy(X2.iloc[valid_index_].values).float()
                    test_x1 = torch.from_numpy(X1.iloc[test_index].values).float()
                    test_x2 = torch.from_numpy(X2.iloc[test_index].values).float()

                    # train ccrl for feature transformation
                    # self.train_ccrl(process_df, train_x1, train_x2, valid_x1, valid_x2, k + 1)
                    self.train_ccrl(process_df, train_x1, train_x2, valid_x1, valid_x2, k + 1, ensem+1)

                    # transformed omics fearures for successive survival type clustering
                    train_x1_all = torch.from_numpy(X1.iloc[c].values).float()
                    train_x2_all = torch.from_numpy(X2.iloc[c].values).float()

                    checkpoint = torch.load(self.model_path + f'/cross_{str(k+1)}model_{str(ensem+1)}_opt_acc.pt')
                    generator = CCRL(self.config_dict['in_dim1'], self.config_dict['in_dim2'], 'cancer_LAML',
                                     self.config_dict['out_dim'], self.config_dict['out_dim'],
                                     task=self.config_dict['task'],
                                     device=self.config_dict['device'])

                    generator.load_state_dict(checkpoint['model'])
                    generator = generator.to(self.config_dict['device'])
                    generator.apply(weight_init)

                    ## out features
                    train_ksi_df, train_eta_df, train_u_df, train_v_df = \
                        self.out_features(generator, train_x1_all.to(self.config_dict['device']),
                                          train_x2_all.to(self.config_dict['device']),
                                          X1.iloc[c].index, X2.iloc[c].index, f'{k+1}_train', ensem+1)
                    test_ksi_df, test_eta_df, test_u_df, test_v_df = \
                        self.out_features(generator, test_x1.to(self.config_dict['device']),
                                          test_x2.to(self.config_dict['device']),
                                          X1.iloc[test_index].index, X2.iloc[test_index].index, f'{k+1}_test', ensem+1)
                    a = 1
                    if survival_label is None:
                        survival_label = self.kmeans_cluster(train_ksi_df, train_eta_df, test_ksi_df, test_eta_df, k + 1)
                        survival_label = survival_label.rename(columns={'label': f'label{ensem}'})
                    else:
                        a = self.kmeans_cluster(train_ksi_df, train_eta_df, test_ksi_df, test_eta_df, k + 1)
                        a = a.rename(columns={'label': f'label{ensem}'})[f'label{ensem}']
                        survival_label = pd.merge(survival_label, a, left_index=True, right_index=True)

                    plt.close()

            survival_label_before = pq.read_table(f'out - 21ensembles/3_cluster_results/trial_{self.trial_num}/survival_label.parquet').to_pandas()
            survival_label_before = survival_label_before.drop(['time', 'event'], axis=1)
            survival_label = pd.merge(survival_label_before, survival_label, left_index=True, right_index=True)
            survival_label['sum'] = survival_label[[f'label{i}' for i in range(ensemble)]].sum(axis=1)
            survival_label['ensemble'] = survival_label['sum'].map(lambda x: 1 if x>=int((ensemble+1)/2) else 0)
            if k == 0:
                self.survival_label = survival_label

            else:
                self.survival_label = pd.concat([self.survival_label, survival_label])

        # save self.survival_label
        pq.write_table(pa.Table.from_pandas(self.survival_label),
                       os.path.join(self.out_address['cluster_results'], 'survival_label.parquet'))
        # Kaplan Meier plot in concatenate validation set (a whole dataset) by using self.survival_label
        self.plot_KM_curve(out_path=self.out_address['KM_curve'])

        # univariate cox-ph model using self.survival_label, the cluster labels as covariate
        self.cox_ph_model()

        # # plot scatter in all test data
        # if len(test_ksi.columns) <= 5:
        #     self.scatter_matrices(test_ksi, test_eta, 'test+all+ksi', 'test_all_eta')
        #     self.scatter_matrices(test_ksi, test_eta, 'test+all+u', 'test_all_u')




    def kmeans_cluster(self, train_ksi: pd.DataFrame, train_eta: pd.DataFrame,
                        test_ksi: pd.DataFrame, test_eta: pd.DataFrame, k: int):
        '''

        :param train_ksi: training torch matrix for view1
        :param train_eta: training torch matrix for view2
        :param test_ksi: testing torch matrix for view1
        :param test_eta: testing torch matrix for view2
        :param k: int, means kth cross-validation
        :return: pd.Dataframe
        '''
        survival = self.config_dict['survival']
        kmeans = KMeans(n_clusters=2).fit(train_ksi.values)
        y_pred_test = kmeans.fit_predict(test_ksi.values)
        survival_label = survival.loc[test_ksi.index, ['time', 'event']]
        survival_label['label'] = y_pred_test
        a = survival_label.groupby('label')['time'].mean()
        if a[0] <= a[1]:
            pass
        else:
            survival_label['label'] = survival_label['label'].apply(lambda x: abs(x-1))

        return survival_label

    def plot_KM_curve(self, out_path):
        label_grouped = self.survival_label.groupby('ensemble')
        fig, ax = plt.subplots(figsize=(20, 16))
        a = {}
        for label, survival in label_grouped:
            a[str(label)] = survival
            kmf1 = KaplanMeierFitter().fit(survival['time'], survival['event'], label=label)
            ax = kmf1.plot_survival_function(ax=ax)
        plt.xlabel('time', fontsize=20)
        plt.ylabel('S(t)', fontsize=20)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, 'Kaplan_Meier_curve_CCRL.png'), dpi=500)
        plt.close()
        self.p_value = logrank_test(a['0']['time'], a['1']['time'], a['0']['event'], a['1']['event']).p_value

    def cox_ph_model(self):
        cph = CoxPHFitter()
        a = self.survival_label[['time', 'event', 'ensemble']]
        # cph.fit(self.survival_label, duration_col='time', event_col='event', strata='label')
        cph.fit(a, duration_col='time', event_col='event')
        print(cph.print_summary())
        self.C_index = cph.concordance_index_

    def train_ccrl(self, process_df, train_x1, train_x2, valid_x1, valid_x2, cross_num, model_num):
        '''

        :param process_df:  record the info of training process
        :param train_x1: training torch matrix for view1
        :param train_x2: training torch matrix for view2
        :param valid_x1: validation torch matrix for view1
        :param valid_x2: validation torch matrix for view2
        :param k:  int, means kth cross-validation
        :return:
        '''
        # model params
        input_dim1 = self.config_dict['in_dim1']
        input_dim2 = self.config_dict['in_dim2']
        task = self.config_dict['task']
        output_dim1 = output_dim2 = self.config_dict['out_dim']
        # output_dim1 = output_dim2 = 50
        class_num = None
        # lr = self.config_dict['lr']
        # wd = self.config_dict['wd']
        lr = 0.001
        wd = 0.0001
        max_epoch = self.config_dict['max_epoch']
        device = self.config_dict['device']
        dataset_name = 'cancer_LAML'
        loss_hyper_param = self.config_dict['loss_hyper_param']

        opt_cor = float("-inf")
        opt_loss = float("+inf")

        # model
        ccrl = CCRL(input_dim1, input_dim2, dataset_name, output_dim1, output_dim2, task=task, class_num=class_num,
                    device=device)
        ccrl = ccrl.to(device)
        ccrl.apply(weight_init)

        # optimizer
        ccrl_optim = optim.Adam(ccrl.parameters(), lr=lr, weight_decay=wd)

        # train and choose the best model (the epoch that has the largest total corr) in validation dataset
        self.model_path = os.path.join(self.out_address['model'], 'cross_'+ str(cross_num), 'model_'+str(model_num))
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        for epoch in range(max_epoch):
            train_loss, train_total_corr = train(ccrl, ccrl_optim, loss_hyper_param,
                                                 train_x1, train_x2, device)
            valid_loss, valid_total_corr = test(ccrl, loss_hyper_param, valid_x1, valid_x2, device)
            print(f'epoch={epoch}, total corr in validation set = {valid_total_corr}')

            # save the intermidiate info of training process
            try:
                process_df.loc[len(process_df)] = [epoch, train_loss.detach().numpy(),
                                                   valid_loss.detach().numpy(),
                                                   train_total_corr, valid_total_corr]
            except:
                process_df.loc[len(process_df)] = [epoch, train_loss.cpu().detach().numpy(),
                                                   valid_loss.cpu().detach().numpy(),
                                                   train_total_corr, valid_total_corr]

            # save the best model
            # self.model_path = self.out_address['model'] + f'/{k}_fold_{dataset_name}_opt_acc.pt'
            # if valid_total_corr > opt_cor:
            #     print(f'Validation cor decreased ({opt_cor:.6f} --> {valid_total_corr:.6f}).  Saving model ...')
            #     state = {'model': ccrl.state_dict(), 'optimizer': ccrl_optim.state_dict(), 'epoch': epoch,
            #              'train_loss': train_loss, 'valid_cor': valid_total_corr}
            #     torch.save(state, self.model_path)
            #     opt_cor = valid_total_corr


            if opt_loss - valid_loss > 1e-3:
                print(f'Validation cor decreased ({opt_loss:.6f} --> {valid_loss:.6f}).  Saving model ...')
                state = {'model': ccrl.state_dict(), 'optimizer': ccrl_optim.state_dict(), 'epoch': epoch,
                         'train_loss': train_loss, 'valid_loss': valid_loss,
                         'train_cor': train_total_corr, 'valid_cor': valid_total_corr}
                torch.save(state, self.model_path +
                           f'/cross_{str(cross_num)}model_{str(model_num)}_opt_acc.pt')
                opt_loss = valid_loss
                count = 0
            else:
                count += 1

            if count >= 40:
                break;

        plt.plot(process_df['epoch'], process_df['train_loss'], label='train_loss')
        plt.plot(process_df['epoch'], process_df['valid_loss'], label='valid_loss')
        plt.xlabel('epoch')
        plt.legend()

        plt.savefig(self.model_path + f'/cross_{str(cross_num)}model_{str(model_num)}_loss_curve.png')
        plt.cla()
        plt.close('all')


        plt.plot(process_df['epoch'], process_df['train_cor'], label='train_cor')
        plt.plot(process_df['epoch'], process_df['valid_cor'], label='valid_cor')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(self.model_path + f'/cross_{str(cross_num)}model_{str(model_num)}_cor_curve.png')
        plt.cla()
        plt.close('all')


        # save the training loss and total corr over every epoch
        path2 = self.model_path + f'/cross_{str(cross_num)}model_{str(model_num)}_training_curves.xlsx'
        process_df.to_excel(path2, index=False)

    def scatter_matrices(self, df1, df2, type1, type2, out_path):
        fig = plt.figure(figsize=(15, 25))
        sns.pairplot(data=df1, vars=df1, diag_kind='kde', markers='+')
        plt.savefig(os.path.join(out_path, type1 + '.png'))


        fig = plt.figure(figsize=(15, 25))
        sns.pairplot(data=df2, vars=df2, diag_kind='kde', markers='+')
        plt.savefig(os.path.join(out_path, type2 + '.png'))


        fig = plt.figure(figsize=(20, 5))
        for i, (col1, col2) in enumerate(zip(df1.columns, df2.columns)):
            plt.subplot(1, df1.shape[1] + 1, i + 1)
            plt.scatter(df1[col1], df2[col2])
        plt.savefig(os.path.join(out_path, f'{type1}_{type2}' + '.png'))


        plt.cla()
        plt.close('all')

    def out_features(self, generator, X1, X2, X1_index, X2_index, cross_num, model_num):
        ksi, eta, u, v = generator(X1, X2)

        out_path = os.path.join(self.out_address['transformed_feas'], 'cross_'+ str(cross_num), 'model_' + str(model_num))
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        ksi_df = pd.DataFrame(ksi.detach().cpu().numpy(), index=X1_index,
                              columns=['omics_ksi_' + str(i) for i in range(1, self.config_dict['out_dim'] + 1)])
        pq.write_table(pa.Table.from_pandas(ksi_df),
                       os.path.join(out_path, 'omics_ksi.parquet'))
        eta_df = pd.DataFrame(eta.detach().cpu().numpy(), index=X2_index,
                              columns=['survival_eta_' + str(i) for i in range(1, self.config_dict['out_dim'] + 1)])
        pq.write_table(pa.Table.from_pandas(eta_df),
                       os.path.join(out_path, 'survival_eta.parquet'))
        u_df = pd.DataFrame(u.detach().cpu().numpy(), index=X1_index,
                            columns=['omics_u_' + str(i) for i in range(1, self.config_dict['out_dim'] + 1)])
        pq.write_table(pa.Table.from_pandas(u_df),
                       os.path.join(out_path, 'omics_u.parquet'))
        v_df = pd.DataFrame(v.detach().cpu().numpy(), index=X2_index,
                            columns=['survival_v_' + str(i) for i in range(1, self.config_dict['out_dim'] + 1)])
        pq.write_table(pa.Table.from_pandas(v_df),
                       os.path.join(out_path, 'survival_v.parquet'))

        # plot scatter
        if len(ksi_df.columns) <=5:
            self.scatter_matrices(ksi_df, eta_df, f'ksi', f'eta', out_path)
            self.scatter_matrices(u_df, eta_df, f'u', f'v', out_path)

        return ksi_df, eta_df, u_df, v_df


if __name__ == "__main__":
    print('pass')
