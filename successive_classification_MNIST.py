import os
import random
import gc
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.svm import SVC
from sklearn import preprocessing

from CCRL import CCRL
from Dataloader import create_datasets
from utils_ import setup_seed


# random seed and device
setup_seed(7)
device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu"))
print(device)


#################TODO: specific ####################
# set params and load pretrained CCRL(TD-CCRL) with task
input_dim1 = 392
input_dim2 = 392
output_dim1 = output_dim2 = 50
task = 'L2R'   # other task: None ,  'L2R' , 'R2L'
class_num = 10

checkpoint = torch.load(r'out/MINIST/L2Ropt_cor.pt')
generator = CCRL(input_dim1, input_dim2, output_dim1, output_dim2, task=task, class_num=class_num,  device=device)
generator.to(device)
generator.load_state_dict(checkpoint['model'])

# load datasets
batch_size = 1000
shuffle = False
train_loader, test_loader,_ = create_datasets(batch_size, shuffle, 'MNIST')
name = rf'out/MINIST/20220903_{task}_epochs=100_opt_cor'
#################TODO: specific ####################

# transformed features

for batch_x1, batch_x2, batch_y in tqdm(train_loader):
    ## to GPU
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
    ## to GPU
    batch_x1 = batch_x1.to(device)
    batch_x2 = batch_x2.to(device)
    batch_y = batch_y.to(device)

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
# load features
train_u = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_train_u.csv')     #(50000, 50)
train_v = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_train_v.csv')     #(50000, 50)
test_u = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_test_u.csv')       #(10000, 50)
test_v = np.loadtxt('GPU_A1B1C1D1_XRMB_2022_08_21_test_v.csv')       #(10000, 50)
# train_x1 = np.loadtxt(os.path.join(name, 'train_x1.csv'))   #(50000, 392)
# train_x2 = np.loadtxt(os.path.join(name, 'train_x2.csv'))   #(50000, 392)
# test_x1 = np.loadtxt(os.path.join(name, 'test_x1.csv'))
# test_x2 = np.loadtxt(os.path.join(name, 'test_x2.csv'))
# train_ksi = np.loadtxt(os.path.join(name, 'train_ksi.csv'))
# train_eta = np.loadtxt(os.path.join(name, 'train_eta.csv'))
# test_ksi = np.loadtxt(os.path.join(name, 'test_ksi.csv'))
# test_eta = np.loadtxt(os.path.join(name, 'test_eta.csv'))
train_y = np.loadtxt(os.path.join(name, 'train_y.csv'))
test_y = np.loadtxt(os.path.join(name, 'test_y.csv'))
################TODO:specific ##################

## calculate total sum correlation
total_sum = 0
total_sum1 = 0
for i in range(output_dim1):
    total_sum = total_sum + pearsonr(test_u[:, i], test_v[:, i])[0]
    total_sum1 = total_sum1 + spearmanr(test_u[:, i], test_v[:, i])[0]

print(f'pearson:{total_sum}')
print(f'spearman:{total_sum1}')


# SVM classification

scaler = preprocessing.StandardScaler().fit(train_u)
train_u = scaler.transform(train_u)
scaler1 = preprocessing.StandardScaler().fit(test_v)
test_v = scaler1.transform(test_v)

clf = SVC(kernel='linear')
clf.fit(train_u, train_y)
c1 = np.sum(clf.predict(test_u) == test_y)/10000
c2 = np.sum(clf.predict(test_v) == test_y)/10000
print('\n',"="*5, "Left test acc：", c1,"="*10,
                  "L2R test acc：", c2, "="*10)
clf1 = SVC(kernel='linear')
clf1.fit(train_v, train_y)
c1 = np.sum(clf1.predict(test_v) == test_y)/10000
c2 = np.sum(clf1.predict(test_u) == test_y)/10000

print('\n',"="*5, "Right test acc：", c1,"="*10,
                  "R2L test acc：", c2, "="*10)

# visualization
plt.scatter(test_u[:,43], test_v[:,43])
plt.xlabel('test_ksi')
plt.ylabel('test_ksi')