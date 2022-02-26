'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import *
import torch
import torch.optim
from torch.utils.data import DataLoader, Dataset 
from torch.optim.lr_scheduler import ExponentialLR
import torchvision
from torchvision import transforms
from fastai.vision.all import *
import gc
import time
import copy

sys.path.append('../util')
from ..util.constants import logger
from ..util.metrics import val_score, get_metrics_fastai
from ..util.optimizer import get_optimizer_fastai
from ..util.loss import get_loss_fastai
from ..util.transforms import get_transforms_cv, update_transforms, get_transforms_fastai

class Model(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.params = dict()
        self.name = ""
        self.id = ""
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def cross_val_score(self, cv=5, scoring='accuracy'):
        pass
    
    @abstractmethod
    def predict(self, y):
        pass
    
    @abstractmethod
    def predict_probas(self, X):
        pass
    
    def get_params(self):
        return self.params

    def get_name(self):
        return self.name
    
    def get_id(self):
        return self.id
    
class SklearnModel(Model):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(SklearnModel, self).__init__(model_type)
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.params = params
        self.model = None
        
    def fit(self):
        model.fit(self.X_train, self.y_test)

    def cross_val_score(self, cv=5, scoring='accuracy'):
        score = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring)
        return score
    
    def predict(self, X: pd.Series):
        return self.model.predict(X)
    
    def predict_probas(self, X: pd.Series):
        return self.model.predict_probas(X)
    
class TsaiModel(Model):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(TsaiModel, self).__init__(model_type)
        self.X = X.to_numpy()
        if self.X.ndim == 2:
            self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
        self.y = y.to_numpy()
        self.params = params
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
        ############## process params #################
        params = self.params.copy()
        self.loss = get_loss_fastai(params.get('loss_func', 'cross_entropy'))
        self.optimizer = get_optimizer_fastai(params.get('opt_func', 'adam'))
        self.metrics = get_metrics_fastai(params.get('metrics', ['accuracy', 'f1_micro', 'recall_micro', 'precision_micro']))
        self.transforms = get_transforms_fastai(params.get('batch_tfms', []))
        self.lr_max = params.get('lr_max', 1e-3)
        self.epochs = params.get('epochs', 25)
        self.splits = TSSplitter(valid_size=0.2, show_plot=False)
        self.bs = params.get('batch_size', [64, 128])
        
        if 'loss_func' in params:
            del params['loss_func']
        if 'opt_func' in params:
            del params['opt_func']
        if 'metrics' in params:
            del params['metrics']
        if 'batch_tfms' in params:
            del params['batch_tfms']
        if 'lr_max' in params:
            del params['lr_max']
        if 'epochs' in params:
            del params['epochs']
        if 'splits' in params:
            del params['splits']
        if 'batch_size' in params:
            del params['batch_size']
        self.params_arch = params
        ############## process params #################
        
        self._init_learner()
            
    def _init_learner(self):
        if self.model_type == 'C':
            learner = TSClassifier
        elif self.model_type == 'R':
            learner = TSRegressor
        elif self.model_type == 'F':
            learner = TSForecaster
        else:
            raise ValueError(f'Unknown type of model: {model_type}')
        self.shuffled_X, self.shuffled_y = shuffle(self.X, self.y) 
        self.splits = TSSplitter()(self.shuffled_X)
        self.learn = learner(self.shuffled_X, self.shuffled_y, bs=self.bs, splits=self.splits, batch_tfms=self.transforms, arch=self.arch, 
                             metrics=self.metrics, arch_config=self.params_arch, device=self.device)
       
    def fit(self):
        self.learn.fit_one_cycle(self.epochs, lr_max=self.lr_max)
                
    def cross_val_score(self, cv=5, scoring='f1'):
        scores = np.zeros(cv)
        for i in range(cv):
            self._init_learner()
            self.fit()
            if scoring == 'accuracy':
                scores[i] = self.learn.recorder.values[-1][2]
            else:
                _, targets, preds = self.learn.get_X_preds(self.shuffled_X[self.splits[1]], self.shuffled_y[self.splits[1]])
                targets = targets.cpu().detach().numpy().tolist()
                preds = [int(p) for p in preds]
                scores[i] = val_score(targets, preds, scoring)
        clear_output()
        return scores
    
    def predict(self, X: pd.Series):
        if X.ndim == 2:
            X = self.X.reshape(X.shape[0], X.shape[1], 1)
        _, _, preds = self.learn.get_X_preds(X)
        return preds
    
    def predict_probas(self, X: pd.Series):
        if X.ndim == 2:
            X = self.X.reshape(X.shape[0], X.shape[1], 1)
        probas, _, _ = self.learn.get_X_preds(X)
        return probas
    
    def plot_metrics(self):
        self.learn.plot_metrics()
        
    def plot_probas(self):
        self.learn.show_probas()
        
    def plot_results(self):
        self.learn.show_results()
        
    def plot_confusion_matrix(self):
        interp = ClassificationInterpretation.from_learner(self.learn)
        interp.plot_confusion_matrix()
    
    def get_params(self):
        params = self.params.copy()
        if 'metrics' in params:
            del params['metrics']
        return params


class PytorchCVModel(Model): 
        
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict()):
        super(PytorchCVModel, self).__init__(model_type)
        self.params = params
        self.dataset = copy.deepcopy(dataset) # WARNING, maybe problematic for large datasets
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        
        ############## process params #################
        params = self.params.copy()
        self.lr_max = params.get('lr_max', 1e-3)
        self.lr_max = 1e-5 # TODO delete
        self.epochs = params.get('epochs', 100)
        self.bs = params.get('batch_size', 16) 
        self.pretrained = params.get('pretrained', False)
        self.transforms = get_transforms_cv(params.get('batch_tfms', []), params=params.get('batch_tfms_params', dict()))
        self.loss = get_loss_fastai(params.get('loss_func', 'cross_entropy'))
        self.optimizer = get_optimizer_fastai(params.get('opt_func', 'adam'))
        self.metrics = get_metrics_fastai(params.get('metrics', ['accuracy', 'f1_micro', 'recall_micro', 'precision_micro']))
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.num_classes = len(dataset.classes)
        else:
            self.num_classes = input('Found no imformation about number of classes. Please enter number of classes: ')
        dataset_shape = dataset[0][0].shape
        self.in_channels = dataset_shape[0]
        self.in_size = dataset_shape[1], dataset_shape[2]
        if params.get('batch_tfms', []) != []:
            self.dataset.transform = self.transforms
        self._check_and_fix_in_size()
        ############## process params #################
        
        self.train_size = int(0.8 * len(dataset))
        self.test_size = len(dataset) - self.train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)  
        self.test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)
     
    @abstractmethod
    def _init_model(self):
        pass

    def _init_learner(self, dataloader):
        self._init_model()
        self.learn = Learner(dataloader, self.model, metrics=self.metrics, loss_func=self.loss, opt_func=self.optimizer)
        self.learn.model.to(self.device)

    def fit(self):
        self.learn.fit_one_cycle(self.epochs, lr_max=self.lr_max)
        
# TODO rework to be consistent with tsai                 
    def evaluate(self):
        self._test(self.test_loader)

    def cross_val_score(self, cv: int=5, scoring: str='f1_micro'):  
        scores = np.zeros(cv)
        for i in range(cv):
            train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.test_size], 
                                                                        generator=torch.Generator().manual_seed(i))
            dataloader = DataLoaders.from_dsets(train_dataset, test_dataset, num_workers=0, device=self.device)
            self._init_learner(dataloader)
            self.fit()
            if scoring == 'accuracy':
                scores[i] = self.learn.recorder.values[-1][2]
            else:
                scores[i] = self._test(dataloader.valid, scoring=scoring)
        clear_output()
        return scores

    def predict(self, X):
        if not torch.is_tensor(X):
            transform = transforms.ToTensor()
            X = transform(X)
        outputs = self.learn.model(X)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().detach().numpy()
    
    def predict_probas(self, X):
        if not torch.is_tensor(X):
            transform = transforms.ToTensor()
            X = transform(X)
        outputs = self.learn.model(X)
        return outputs.cpu().detach().numpy()

    def _test(self, test_loader: DataLoader, scoring: str='f1_micro'):
        correct = 0
        total = 0
        preds = []
        targets = []
        self.model.eval()
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for p in predicted:
                    preds.append(p.cpu().detach().numpy().tolist())
                for t in labels:
                    targets.append(t.cpu().detach().numpy().tolist())  
        if scoring == 'accuracy':
            score = correct / total
        else:
            score = val_score(targets, preds, scoring)  
        return score                       
    
    def _check_and_fix_in_size(self):    
        if 'resize' not in self.params.get('batch_tfms', []) and self.in_size not in self.std_in_sizes:
            logger.info(f"No transformation given. Resize images to standard input size of: {self.name}")
            self.in_size = min(self.std_in_sizes, key=lambda x: abs(x[0]- self.in_size[0]) + abs(x[1]- self.in_size[1]))
            new_transforms = [transforms.Resize(self.in_size)]
            update_transforms(self.dataset, new_transforms)
        
    def get_params(self):
        params = self.params.copy()
        if 'metrics' in params:
            del params['metrics']
        return params
    