'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from PIL import Image
import io
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .constants import logger
from .transforms import *

TS_DATASETS = ['iris', 'wine', 'diabetes', 'breast_cancer']
CV_DATASETS = ['mnist', 'fashion_mnist', 'cifar', 'celeba']


def load_dataset(name: str, shuffle: bool=False, split: bool=False, transform: Any=None, transform_params: Dict=None):
    if name in TS_DATASETS:
        return _load_ts_dataset(name, shuffle=shuffle, split=split, transform=transform)
    elif name in CV_DATASETS: 
        return _load_cv_dataset(name, shuffle=shuffle, split=split, transform=transform, transform_params=transform_params)
    else:
        logger.error('Unknown dataset')

def _load_ts_dataset(name: str, shuffle: bool=False, split: bool=False, transform: Dict=None):
    if name == 'iris':
        dataset = load_iris()
    elif name == 'wine':
        dataset = load_wine()
    elif name == 'diabetes':
        dataset = load_diabetes()
    elif name == 'breast_cancer':
        dataset = load_breast_cancer()
    else:
        logger.error('Unknown dataset')
        
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = pd.Series(dataset.target)
    
    if transform:
        df = apply_transforms(df, transform)
    
    if shuffle:
        df = sklearn.utils.shuffle(df)
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    if split:
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, y_train, X_test, y_test
    else:
        return X, y

def _load_cv_dataset(name: str, shuffle: bool=False, split: bool=False, transform: List=None, transform_params: Dict=None):
    if name == 'mnist':
        if transform:
            transform = get_transforms_cv(transform, transform_params)
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = torchvision.datasets.MNIST('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.MNIST('../../data', train=False, transform=transform, download=True) 
    elif name == 'fashion_mnist':
        if transform:
            transform = get_transforms_cv(transform, transform_params)
        else:
            transform=transforms.Compose([transforms.ToTensor()])
        dataset_train = torchvision.datasets.FashionMNIST('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.FashionMNIST('../../data', train=False, transform=transform, download=True) 
    elif name == 'cifar':
        if transform:
            transform = get_transforms_cv(transform, transform_params)
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = torchvision.datasets.CIFAR10('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.CIFAR10('../../data', train=False, transform=transform, download=True) 
    elif name == 'celeba':
        if transform:
            transform = get_transforms_cv(transform, transform_params)
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = torchvision.datasets.CelebA('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.CelebA('../../data', train=False, transform=transform, download=True) 
    else:
        logger.error('Unknown dataset')
    
    if split:
        X_train, y_train = convert_dataset_to_numpy(dataset_train, shuffle=shuffle)
        X_test, y_test = convert_dataset_to_numpy(dataset_test, shuffle=shuffle)
        return X_train, y_train, X_test, y_test
    else:
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
        X, y = convert_dataset_to_numpy(dataset, shuffle=shuffle)
        return X, y

def convert_dataset_to_numpy(dataset, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=shuffle)
    X, y = next(iter(dataloader))
    X = X.numpy() 
    y = y.numpy()
    return X, y    
    
################## Util datasets ############################   
    
class NumpyDataset(Dataset):
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            X = Image.fromarray(self.data[idx].astype(np.uint8).transpose(1,2,0))
            X = self.transform(X)
        
        return X, y
    
    def __len__(self):
        return len(self.data)
    
class CSVFolderDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Keyword arguments:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        img_name = os.path.join(self.root_dir, self.df.loc[idx, 'name'])
        X = io.imread(img_name)
        y = self.df.loc[idx, 'target']

        if self.transform:
            X = self.transform(X)

        return X, y
    
class CSVDataset(Dataset):
    
    # TODO Mode necessary?
    def __init__(self, csv_file, transform=None, mode='gray'):
        """
        Keyword arguments:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # TODO add standard implementation for csv (rgb and gray)
        if 'width' in self.df and 'heigth' in self.df:
            width = self.df.loc[idx, 'width']
            heigth = self.df.loc[idx, 'heigth']
        elif 'size' in self.df:
            width = self.df.loc[idx, 'size']
            heigth = self.df.loc[idx, 'size']
        elif 'width' in self.df: 
            width = self.df.loc[idx, 'width']
            heigth = (self.df.shape[1] - 1) // width
        elif 'heigth' in self.df:
            heigth = self.df.loc[idx, 'heigth']
            width = (self.df.shape[1] - 1) // heigth
        else:
            width = (self.df.shape[1] - 1) // 2
            heigth = width
        
        array = np.array((width, heigth))
        for y in range(heigth):
            for x in range(width):
                array[x, y] = self.df.loc[idx, 'pix-' + str(x) + str(y)]        
        X = Image.fromarray(array)                
        if self.transform:
            X = self.transform(X)
        
        if 'target' in self.df:
            y = self.df.loc[idx, 'target']
        elif 'label' in self.df:
            y = self.df.loc[idx, 'label']
        else:
            logger.error('Target not found')
        return X, y   

class Data_rb_cla(Dataset):
    
    def __init__(self, Xs, ys):
        # input is type of pandas
        self.Xs = torch.from_numpy(Xs.to_numpy()).float()
        self.ys = torch.from_numpy(ys.to_numpy()).long()

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]


class Data_rb_reg(Dataset):
    
    def __init__(self, Xs, ys):
        # input is type of pandas
        self.Xs = torch.from_numpy(Xs.to_numpy()).float()
        self.ys = torch.from_numpy(ys.to_numpy()).float()

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]
    
class MLP(torch.nn.Module):
    
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.ln = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        '''
        inp shape of torch tensor
        '''
        out = self.ln(self.layers(inp))
        return out