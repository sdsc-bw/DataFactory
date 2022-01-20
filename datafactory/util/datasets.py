import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.utils
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from .constants import logger

def load_ts_dataset(name, shuffle=True, split=False):
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
    df['class'] = pd.Series(dataset.target)
        
    if shuffle:
        df = sklearn.utils.shuffle(df)
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    if split:
        return train_test_split(X, y, test_size=0.2)
    else:
        return X, y

def load_cv_dataset(name, split=False):
    transform = transforms.Compose([transforms.ToTensor()])
    if name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = torchvision.datasets.MNIST('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.MNIST('../../data', train=False, transform=transform, download=True) 
    elif name == 'fashion_mnist':
        transform=transforms.Compose([transforms.ToTensor()])
        dataset_train = torchvision.datasets.FashionMNIST('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.FashionMNIST('../../data', train=False, transform=transform, download=True) 
    elif name == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = torchvision.datasets.CIFAR10('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.CIFAR10('../../data', train=False, transform=transform, download=True) 
    elif name == 'celeba':
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = torchvision.datasets.CelebA('../../data', train=True, transform=transform, download=True)
        dataset_test = torchvision.datasets.CelebA('../../data', train=False, transform=transform, download=True) 
    else:
        logger.error('Unknown dataset')
    
    if split:
        dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train))
        X_train = next(iter(dataloader_train))[0].numpy()
        y_train = next(iter(dataloader_train))[1].numpy()
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
        X_test = next(iter(dataloader_test))[0].numpy()
        y_test = next(iter(dataloader_test))[1].numpy()
        return X_train, y_train, X_test, y_test
    else:
        dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        X = next(iter(dataloader))[0].numpy()
        y = next(iter(dataloader))[1].numpy()
        return X, y
        