# coding=utf-8
# @FileName  :CCRL.py
# @Time      :2022/8/2 12:41
# @Author    :Hou Jiaqi

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# global vatiables
id_column = 'SampleID'


class PreProcess():
    '''
    1. merge data matrices by patients
    2. out the statistics of the data
    3. calculate HAZ(t) and S(t) according to the survival info (time and event),
       and add the calculated two features to survival info (time, event, HAZ, S),
       the later one called survival_improved
    '''

    def __init__(self, address_dic: dict):
        # attribution
        self.data_dict = {}
        for key, value in address_dic.items():
            self.data_dict[key] = pd.read_table(address_dic[key])
        self.data_dict['survival'] = self.data_dict['survival'].rename(columns={'sample': 'SampleID'})
        #self.data_dict['survival_improved'] = self.data_dict['survival']
        self.statistic_table = None
        self.HAZ_dict = {}
        self.S_dict = {}

        # method
        self.data_merge()
        self.calculate_HAZ_and_S()
        self.create_features4survival()

    def data_merge(self):
        # records
        self.statistic_table = pd.DataFrame(columns=['View', 'NumOfFeatures', 'NumOfPreSamples', 'NumOfMergedSamples'])
        self.statistic_table['View'] = list(self.data_dict.keys())
        self.statistic_table.set_index('View', inplace=True)

        # identify the valid patients who are with multi-view data (4-view)
        for i, (key, value) in enumerate(self.data_dict.items()):
            if i == 0:
                patients = self.data_dict[key][id_column]
            else:
                patients = pd.merge(patients, self.data_dict[key], on=id_column, how='inner')
                patients = patients[id_column]

        # filter the invalid patients and any-nan columns
        patients_list = list(patients)
        for key, value in self.data_dict.items():
            self.data_dict[key].set_index(id_column, inplace=True)
            # records
            self.statistic_table.loc[key, 'NumOfPreSamples'] = len(self.data_dict[key])

            self.data_dict[key] = self.data_dict[key].loc[patients_list]
            self.data_dict[key].dropna(axis=1, how='any', inplace=True)

            # records
            self.statistic_table.loc[key, 'NumOfMergedSamples'] = len(self.data_dict[key])
            self.statistic_table.loc[key, 'NumOfFeatures'] = len(self.data_dict[key].columns)

    def calculate_HAZ_and_S(self):
        df = self.data_dict['survival'].sort_values(by='time')
        time_list = df['time'].unique()
        self.HAZ_dict['0'] = 0
        self.S_dict['0'] = 1
        last_S = self.S_dict['0']  # S of the last time
        for time in time_list:
            a = df[df['time'] >= time]
            condition = (a['event'] == 0) & (a['time'] == time)
            a = a.mask(condition)
            a.dropna(axis=0, how='all', inplace=True)
            risked = len(a)
            b = a[a['time']==time]
            died = len(b)
            if risked == 0:
                pass
            else:
                self.HAZ_dict[time] = died/risked
                self.S_dict[time] = last_S * (1-self.HAZ_dict[time])
                # update last S
                last_S = self.S_dict[time]

    def create_features4survival(self):
        df = self.data_dict['survival'].copy()
        df['HAZ'] = df['time'].map(self.HAZ_dict)
        df['S'] = df['time'].map(self.S_dict)

        # delete the sample with no HAZ and S
        self.data_dict['survival_improved'] = df.dropna(axis=0, how='any')
        patients = self.data_dict['survival_improved'].index.tolist()
        for key in self.data_dict.keys():
            self.data_dict[key] = self.data_dict[key].loc[patients]

        # update record
        self.statistic_table['NumOfMergedSamples'] = len(patients)


if __name__ == "__main__":
    # data configuration
    """
    Processed LAML data from: https://doi.org/10.6084/m9.figshare.14832813.v1
    """
    config_dict = {'METH': 'LAML_matrices/mapped_METH.tsv',
                   'MIR': 'LAML_matrices/mapped_MIR.tsv',
                   'RNA': 'LAML_matrices/mapped_RNA.tsv',
                   'survival': 'LAML_matrices/survival_laml.tsv'}
    process = PreProcess(config_dict)

    # out data, parquet format
    concat_omics = pd.merge(process.data_dict['METH'], process.data_dict['MIR'],
                            left_index=True, right_index=True, suffixes=('_METH', '_MIR'))
    concat_omics = pd.merge(concat_omics, process.data_dict['RNA'],
                            left_index=True, right_index=True, suffixes=('', '_RNA'))

    # Z_score normalization for concat_omics
    omic_std = concat_omics.std()
    col_list = omic_std[omic_std != 0].index.to_list()
    concat_omics = concat_omics[col_list]
    concat_omics = (concat_omics - concat_omics.mean()) / concat_omics.std()

    survival_base = process.data_dict['survival']
    survival_improved = process.data_dict['survival_improved']
    # ##TODO: attention
    # survival_base['stand_time'] = (survival_base['time'] - survival_base['time'].min()) / (survival_base['time'].max()-survival_base['time'].min())
    # survival_improved['stand_time'] = survival_base['time']
    # pq.write_table(pa.Table.from_pandas(concat_omics), 'data/df_concat_omics.parquet')
    # pq.write_table(pa.Table.from_pandas(survival_base), 'data/df_survival_base.parquet')
    # pq.write_table(pa.Table.from_pandas(survival_improved), 'data/df_survival_improved.parquet')
    process.statistic_table.to_excel('data/statistic_table.xlsx')



