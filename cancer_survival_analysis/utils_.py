import gzip
import numpy as np
import torch
import struct
import random


def load_image(data_file):
    '''
    load images
    '''
    print('loading image ...')
    with gzip.open(data_file, 'rb') as f:
        #r means read, b is binary
        magic, size = struct.unpack(">II",f.read(8))
        # print('magic:', magic)  ##2051
        # print('size:', size)    ##60000 or 10000
        nrows, ncols = struct.unpack(">II",f.read(8))
        # print('nrows:', nrows)  28
        # print('ncols:', ncols)  28
        #print('=====================')
        data = np.frombuffer(f.read(),dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
        return data


def load_label(data_file):
    '''
    load labels of images
    '''
    print('loading label ...')
    with gzip.open(data_file, 'rb') as f:
        #r means read, b is binary
        magic, size = struct.unpack(">II",f.read(8))
        # print('magic:', magic)  ##2049
        # print('size:', size)    ##60000 or 10000
        #print('=====================')
        data = np.frombuffer(f.read(),dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, 1))
        return data


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True