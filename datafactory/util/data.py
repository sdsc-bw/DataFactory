import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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