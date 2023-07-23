import os
import gc
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing

from CCRL import CCRL
from Dataloader import create_datasets
from utils_ import setup_seed


# random seed and device
setup_seed(7)
device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu"))
print(device)

###########TODO:specific##################
# set params and load pretrained TD-CCRL with 'SupervisedDataFusion' task
input_dim1 = 273
input_dim2 = 112
output_dim1 = output_dim2 = 112
task = 'SupervisedDataFusion'   # other task: None ,  'L2R' , 'R2L'
class_num = 40

checkpoint = torch.load(r'out/XRMB/SupervisedDataFusionopt_acc.pt')
generator = CCRL(input_dim1, input_dim2, output_dim1, output_dim2, task=task, class_num=class_num,  device=device)
generator.to(device)
generator.load_state_dict(checkpoint['model'])

# load train data
batch_size = 1000
shuffle = False
train_loader, test_loader,_ = create_datasets(batch_size, shuffle, 'XRMB')
name = rf'out/XRMB/20221003_{task}_epochs=100_opt_acc'  # path for save features
###########TODO:specific##################

# feature transformation by pretrained TD-CCRL

for batch_x1, batch_x2, batch_y in tqdm(train_loader):
    ##  to GPU
    batch_x1 = batch_x1.to(device)
    batch_x2 = batch_x2.to(device)
    batch_y = batch_y.to(device)
    ## out features
    if task is None:
        ksi, eta, u, v = generator(batch_x1, batch_x2)
    else:
        ksi, eta, u, v, _ = generator(batch_x1, batch_x2)

    with open(os.path.join(name, 'train_x1.csv'), 'ab') as f1:
        np.savetxt(f1, batch_x1.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'train_x2.csv'), 'ab') as f2:
        np.savetxt(f2, batch_x2.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'train_ksi.csv'), 'ab') as f3:
        np.savetxt(f3, ksi.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'train_eta.csv'), 'ab') as f4:
        np.savetxt(f4, eta.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'train_y.csv'), 'ab') as f5:
        np.savetxt(f5, batch_y.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'train_U.csv'), 'ab') as f6:
        np.savetxt(f6, u.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'train_V.csv'), 'ab') as f7:
        np.savetxt(f7, v.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

for batch_x1, batch_x2, batch_y in tqdm(test_loader):
    ## to gpu
    batch_x1 = batch_x1.to(device)
    batch_x2 = batch_x2.to(device)
    batch_y = batch_y.to(device)

    ## out features
    if task is None:
        ksi, eta, u, v = generator(batch_x1, batch_x2)
    else:
        ksi, eta, u, v, _ = generator(batch_x1, batch_x2)

    with open(os.path.join(name, 'test_x1.csv'), 'ab') as f1:
        np.savetxt(f1, batch_x1.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'test_x2.csv'), 'ab') as f2:
        np.savetxt(f2, batch_x2.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'test_ksi.csv'), 'ab') as f3:
        np.savetxt(f3, ksi.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'test_eta.csv'), 'ab') as f4:
        np.savetxt(f4, eta.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'test_y.csv'), 'ab') as f5:
        np.savetxt(f5, batch_y.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'test_U.csv'), 'ab') as f6:
        np.savetxt(f6, u.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(name, 'test_V.csv'), 'ab') as f7:
        np.savetxt(f7, v.detach().cpu().numpy())
        gc.collect()
        torch.cuda.empty_cache()

################TODO:specific ##################
## load transformed features
train_x1 = np.loadtxt(os.path.join(name, 'train_x1.csv'))   #(1429236, 273)
train_x2 = np.loadtxt(os.path.join(name, 'train_x2.csv'))   #(1429236, 112)
test_x1 = np.loadtxt(os.path.join(name, 'test_x1.csv'))
test_x2 = np.loadtxt(os.path.join(name, 'test_x2.csv'))
train_ksi = np.loadtxt(os.path.join(name, 'train_ksi.csv'))
train_eta = np.loadtxt(os.path.join(name, 'train_eta.csv'))
test_ksi = np.loadtxt(os.path.join(name, 'test_ksi.csv'))
test_eta = np.loadtxt(os.path.join(name, 'test_eta.csv'))
train_y = np.loadtxt(os.path.join(name, 'train_y.csv'))
test_y = np.loadtxt(os.path.join(name, 'test_y.csv'))
# train_u = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_train_u.csv')     #(1429236, 112)
# train_v = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_train_v.csv')     #(1429236, 112)
# test_u = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_test_u.csv')       #(111314, 112)
# test_v = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_test_v.csv')       #(111314, 112)
###########TODO:specific##################

## calculate total sum correlation
total_sum = 0
total_sum1 = 0
for i in range(output_dim1):
    total_sum = total_sum + pearsonr(test_ksi[:, i], test_eta[:, i])[0]
    total_sum1 = total_sum1 + spearmanr(test_ksi[:, i], test_eta[:, i])[0]

print(f'pearson:{total_sum}')
print(f'spearman:{total_sum1}')

# successive classification by using LDA

# classification using concatenate transformed features
clf = LDA()
train = np.concatenate((train_ksi, train_eta), axis=1)
test = np.concatenate((test_ksi, test_eta), axis=1)

scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)
scaler1 = preprocessing.StandardScaler().fit(test)
test = scaler1.transform(test)

clf.fit(train, train_y)
c1 = np.sum(clf.predict(train) == train_y)/train.shape[0]
c2 = np.sum(clf.predict(test) == test_y)/test.shape[0]
print('\n', "="*5, "training acc：", c1, "="*10, "tesing acc：", c2, "="*10)


# classification using concatenate original and transformed features
clf1 = LDA(solver='svd')
train = np.concatenate((train_x1, train_x2, train_ksi, train_eta), axis=1)
test = np.concatenate((test_x1, test_x2, test_ksi, test_eta), axis=1)

scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)
scaler1 = preprocessing.StandardScaler().fit(test)
test = scaler1.transform(test)

clf1.fit(train, train_y)
c1 = np.sum(clf1.predict(train) == train_y)/train.shape[0]
c2 = np.sum(clf1.predict(test) == test_y)/test.shape[0]
print('\n', "="*5, "training acc：", c1, "="*10, "testing acc：", c2, "="*10)

# classification using concatenate original features
clf2 = LDA()

train = np.concatenate((train_x1, train_x2), axis=1)
test = np.concatenate((test_x1, test_x2), axis=1)

scaler = preprocessing.StandardScaler().fit(train)
train=scaler.transform(train)
scaler1 = preprocessing.StandardScaler().fit(test)
test=scaler1.transform(test)

clf2.fit(train, train_y)

c1 = np.sum(clf2.predict(train) == train_y)/train.shape[0]
c2 = np.sum(clf2.predict(test) == test_y)/test.shape[0]

print('\n',"="*5, "training acc：", c1,"="*10, "testing acc：", c2,"="*10)

# visualize
plt.scatter(test_ksi[:,43], test_eta[:,43])
plt.xlabel('test_ksi')
plt.ylabel('test_ksi')