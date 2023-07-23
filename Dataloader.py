# coding=utf-8
# @FileName  :Dataloader.py
# @Time      :2022/8/5 9:00
import numpy as np
import scipy.io
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

from utils_ import load_image, load_label


def create_datasets(batch_size, shaffle, string):

    if string == 'XRMB':
        # laod XRMB datasets(two views):https://home.ttic.edu/~klivescu/XRMB_data/full/README
        mat1 = scipy.io.loadmat(r'XRMBf2KALDI_window7_single1.mat')
        print('mat1.keys:', mat1.keys())
        mat2 = scipy.io.loadmat(r'XRMBf2KALDI_window7_single2.mat')
        print('mat2.keys:', mat2.keys())

        # load train, val, test data
        train_x1 = mat1['X1']  # <class 'numpy.ndarray'>  (1429236, 273) float32
        train_x2 = mat2['X2']  # <class 'numpy.ndarray'>  (1429236, 112) float32
        train_y_np = mat2['trainLabel'].squeeze()  # <class 'numpy.ndarray'>  (1429236,) uint8

        val_x1 = mat1['XV1']  # <class 'numpy.ndarray'>  (85297, 273) float32
        val_x2 = mat2['XV2']  # <class 'numpy.ndarray'>  (85297, 112) float32
        val_y_np = mat2['tuneLabel'].squeeze()  # <class 'numpy.ndarray'>  (85297,) uint8

        test_x1 = mat1['XTe1']  # <class 'numpy.ndarray'>  (111314, 273) float32
        test_x2 = mat2['XTe2']  # <class 'numpy.ndarray'>  (111314, 112) float32
        test_y_np = mat2['testLabel'].squeeze()  # <class 'numpy.ndarray'>  (111314,) uint8

        del mat1, mat2

    elif string == 'MNIST':

        # load MINIST datasets from Lecun1998 : http://yann.lecun.com/exdb/mnist/
        train_x_ = load_image(r'train-images-idx3-ubyte.gz').astype(
            float)  # <class 'numpy.ndarray'> (60000, 28, 28) float64(before Uint8)
        train_y_ = load_label(r'train-labels-idx1-ubyte.gz').astype(
            int).squeeze()  # <class 'numpy.ndarray'> (60000, ) int64
        test_x = load_image(r't10k-images-idx3-ubyte.gz').astype(
            float)  # <class 'numpy.ndarray'> (10000, 28, 28) float64
        test_y_np = load_label(r't10k-labels-idx1-ubyte.gz').astype(
            int).squeeze()  # <class 'numpy.ndarray'>(10000, ) int64

        train_x, val_x, train_y_np, val_y_np = train_test_split(train_x_, train_y_, test_size=1 / 6)

        train_x1 = np.array([dt[:, :14].flatten() for dt in train_x]) # <class 'numpy.ndarray'>(50000, 392)
        train_x2 = np.array([dt[:, 14:].flatten() for dt in train_x]) # <class 'numpy.ndarray'>(50000, 392)

        val_x1 = np.array([dt[:, :14].flatten() for dt in val_x])  # <class 'numpy.ndarray'>(10000, 392)
        val_x2 = np.array([dt[:, 14:].flatten() for dt in val_x])  # <class 'numpy.ndarray'>(10000, 392)

        test_x1 = np.array([dt[:, :14].flatten() for dt in test_x])  # <class 'numpy.ndarray'>(10000, 392)
        test_x2 = np.array([dt[:, 14:].flatten() for dt in test_x])  # <class 'numpy.ndarray'>(10000, 392)

    else:
        exec('incompatible datasets')

    # data to tensor, default:torch.float=torch.float32
    train_x1 = torch.tensor(train_x1, dtype=torch.float)  # torch.Size([1429236, 273]) torch.float32
    train_x2 = torch.tensor(train_x2, dtype=torch.float)   # torch.Size([1429236, 112]) torch.float32
    train_y = torch.tensor(train_y_np, dtype=torch.float)  # torch.Size([1429236]) torch.float32

    val_x1 = torch.tensor(val_x1, dtype=torch.float)  # torch.Size([85297, 273]) torch.float32
    val_x2 = torch.tensor(val_x2, dtype=torch.float)  # torch.Size([85297, 112]) torch.float32
    val_y = torch.tensor(val_y_np, dtype=torch.float)  # torch.Size([85297]) torch.float32

    test_x1 = torch.tensor(test_x1, dtype=torch.float)  # torch.Size([111314, 273]) torch.float32
    test_x2 = torch.tensor(test_x2, dtype=torch.float)  # torch.Size([111314, 112]) torch.float32
    test_y = torch.tensor(test_y_np, dtype=torch.float)  # torch.Size([111314]) torch.float32

    # dataloader
    train_loader = Data.DataLoader(Data.TensorDataset(train_x1, train_x2, train_y),
                                   batch_size=batch_size,
                                   shuffle=shaffle)

    valid_loader = Data.DataLoader(Data.TensorDataset(val_x1, val_x2, val_y),
                                   batch_size=batch_size,
                                   shuffle=shaffle)

    test_loader = Data.DataLoader(Data.TensorDataset(test_x1, test_x2, test_y),
                                  batch_size=batch_size,
                                  shuffle=shaffle)

    return train_loader, test_loader, valid_loader
