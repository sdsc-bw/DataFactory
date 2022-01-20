'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NumpyDataset(Dataset):
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

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