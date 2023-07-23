# coding=utf-8
# @Time    : 2023/3/6 17:00
# @Author  : Hou Jiaqi
# @File    : trial_design.py
# @Project : CCRL_cancer_survival_clustering
import os.path

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq


# from obtain_trial_index import ObtainTrialIndex
from sklearn.model_selection import KFold

from obtain_trial_index_2 import ObtainTrialIndex
from utils_ import setup_seed


class TrialDesigner():
    '''
    1. considering the influence of model(use survival_base or survival_improved),
        out dimensions of transformed features(10, 50, 100),
        learning rate weight decay in Adam optimizer(0.001--0, 0.001--0.0001, 0.0001--0) on out features.
    2. utilizing Orthogonal Design Theory, set the mentioned one two-level factor and two three-level factors in an
        L_3^9 orthogonal table by "draws up the level method",  variance analysis to identify significant factors and
        optimal levels.
    3. p-value(the smaller, the better) and C-index(the closer to 1, the better) as test-index of orthogonal design and
        variance analysis: we get the transformed omics features of CCRL under every trial, then cluster them to 2
        groups using k-means clustering. The p-value of log-rank test for testing if the two groups' Kaplan-Meier curve
        are significant differences, C-index of a Cox-PH model by using cluster labels as univariate covariate to verify
        the effectiveness of the cluster label.
    '''

    def __init__(self, in_address_dict: dict, out_address_dict: dict, fold_num: int, task=None):
        # attribution
        self.concat_omics = pq.read_table(in_address_dict['concat_omics']).to_pandas()
        self.survival_base = pq.read_table(in_address_dict['survival_base']).to_pandas()
        self.survival_improved = pq.read_table(in_address_dict['survival_improved']).to_pandas()
        self.out_address_dict = out_address_dict
        self.factor_level_table = None
        self.orthogonal_table = None
        self.fold_num = fold_num
        self.task = task

        # method
        self.set_factor_level_table()
        self.set_orthogonal_table()
        self.auto_trial()

    def set_factor_level_table(self):
        self.factor_level_table = pd.DataFrame(columns=['FactorMark', 'Factors', 'Level1', 'Level2', 'Level3'])
        self.factor_level_table['FactorMark'] = ['A', 'B', 'C']
        self.factor_level_table['Factors'] = ['model', 'out_dim', 'lr-wd']
        self.factor_level_table['Level1'] = ['improved', '5', '0.001-0']
        self.factor_level_table['Level2'] = ['base', '20', '0.001-0.0001']
        self.factor_level_table['Level3'] = [np.nan, '50', '0.0001-0']
        self.factor_level_table.set_index('FactorMark', inplace=True)

    def set_orthogonal_table(self):
        # set L_3^9 orthogonal table
        self.orthogonal_table = pd.DataFrame(columns=['TrialNum', 'A1_', '1', 'B2', 'C3', '4', 'y_pvalue', 'y_Cindex'])
        self.orthogonal_table['TrialNum'] = [i for i in range(1, 10)]
        self.orthogonal_table['A1_'] = [1, 1, 1,
                                        2, 2, 2,
                                        2, 2, 2]
        self.orthogonal_table['1'] = [1, 1, 1,
                                      2, 2, 2,
                                      3, 3, 3]
        self.orthogonal_table['B2'] = [1, 2, 3,
                                       1, 2, 3,
                                       1, 2, 3]
        self.orthogonal_table['C3'] = [1, 2, 3,
                                       2, 3, 1,
                                       3, 1, 2]
        self.orthogonal_table['4'] = [1, 2, 3,
                                      3, 1, 2,
                                      2, 3, 1]
        self.orthogonal_table.set_index('TrialNum', inplace=True)

    def set_config_dict(self, trial_num: int):
        config_dict = {'fold_num': self.fold_num,
                       'task': self.task,
                       'concat_omics': self.concat_omics,
                       'survival': None,
                       'in_dim1': len(self.concat_omics.columns),
                       'in_dim2': None,
                       'out_dim': None,
                       'lr': None,
                       'wd': None,
                       'max_epoch': 500,
                       'loss_hyper_param': {'a1': 1, 'a2': 1/3, 'a3': None},
                       'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}

        # set survival features and in_dim1
        col = 'Level' + str(self.orthogonal_table.loc[trial_num, 'A1_'])
        a = self.factor_level_table.loc['A', col]
        if a == 'base':
            config_dict['survival'] = self.survival_base
        elif a == 'improved':
            config_dict['survival'] = self.survival_improved
        else:
            raise Exception("please set the matched self.factor_level_table and self.orthogonal_table")
        config_dict['in_dim2'] = len(config_dict['survival'].columns)

        #  set out dimension
        col = 'Level' + str(self.orthogonal_table.loc[trial_num, 'B2'])
        try:
            config_dict['out_dim'] = int(self.factor_level_table.loc['B', col])
        except:
            raise Exception("please set the matched self.factor_level_table and self.orthogonal_table")

        #  set lr and wd in Adam optimizer
        col = 'Level' + str(self.orthogonal_table.loc[trial_num, 'C3'])
        try:
            config_dict['lr'] = float(self.factor_level_table.loc['C', col].split('-')[0])
            config_dict['wd'] = float(self.factor_level_table.loc['C', col].split('-')[1])
        except:
            raise Exception("please set the matched self.factor_level_table and self.orthogonal_table")

        # set a3
        config_dict['loss_hyper_param']['a3'] = 1/(config_dict['out_dim']-1)
        return config_dict

    def auto_trial(self):
        for i in self.orthogonal_table.index.to_list():
            if i<9:
                k_fold = KFold(n_splits=5, shuffle=True, random_state=7)
                for k, (train_index, test_index) in enumerate(k_fold.split([0]*145)):
                    # 集成模型，使用训练集中的80%
                    for ensem in range(31):
                        c = train_index.copy()
                        np.random.shuffle(c)
                        if ensem <= 20:
                            pass
            else:
                # set config dict for every trial
                config_dict = self.set_config_dict(i)
                # train CCRL and save model
                address_dict = {}
                for key in self.out_address_dict.keys():
                    address_dict[key] = os.path.abspath('.') + '/' + self.out_address_dict[key] + f'/trial_{i}'
                    if not os.path.exists(address_dict[key]):
                        os.makedirs(address_dict[key])
                trail_class = ObtainTrialIndex(config_dict, address_dict, i)
                # trail_class = ObtainTrialIndex(config_dict, address_dict)
                trail_class.cross_validation()
                self.orthogonal_table.loc[i, 'y_pvalue'] = trail_class.p_value
                self.orthogonal_table.loc[i, 'y_Cindex'] = trail_class.C_index

                self.orthogonal_table.to_excel(os.path.join(self.out_address_dict['orthog_trail'], 'othog_table.xlsx'))

                # obtain transformed multi-omics features
                # succesive task, including survival level clustering, K-M survival curve, univariate cox-ph model
                #

    # def variance_analysis(self):


if __name__ == "__main__":

    setup_seed(7)
    in_address_dict = {'concat_omics': 'data/df_concat_omics.parquet',
                       'survival_base': 'data/df_survival_base.parquet',
                       'survival_improved': 'data/df_survival_improved.parquet'}

    out_address_dict = {'model': 'out/1_model',
                        'transformed_feas': 'out/2_transformed_features',
                        'cluster_results': 'out/3_cluster_results',
                        'KM_curve': 'out/4_KM_curve',
                        'orthog_trail': 'out/5_orthogonal_trial'}

    trail_designer = TrialDesigner(in_address_dict, out_address_dict, fold_num=5)
