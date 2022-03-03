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
from ..util.metrics import get_metrics_fastai, contvert_to_sklearn_metrics
from ..util.optimizer import get_optimizer_fastai
from ..util.loss import get_loss_fastai
from ..util.transforms import get_transforms_cv, update_transforms, get_transforms_ts

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
    def cross_val_score(self, cv=5, scoring='f1_micro'):
        pass
    
    @abstractmethod
    def predict(self, y):
        pass
    
    @abstractmethod
    def predict_probas(self, X):
        pass
    
    def get_params(self):
        params = self.params.copy()
        return params

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

    def cross_val_score(self, cv=5, scoring='f1_micro'):
        scoring_temp = contvert_to_sklearn_metrics(scoring)
        score = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring_temp)
        if scoring == 'mse' or scoring == 'mae':
            return -score
        else:
            return score
    
    def predict(self, X: pd.Series):
        return self.model.predict(X)
    
    def predict_probas(self, X: pd.Series):
        return self.model.predict_probas(X)

class FastAIModel(Model):
    
    def __init__(self, model_type: str, params:Dict=dict(), metric_average='micro', device='gpu'):
        super(FastAIModel, self).__init__(model_type)
        
        if device == 'gpu':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif device == 'cpu':
                    self.device = torch.device('cpu')
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        self.params = params
        ############## process params #################
        params = self.params.copy()
        self.loss = get_loss_fastai(params.get('loss_func', 'cross_entropy'))
        self.optimizer = get_optimizer_fastai(params.get('opt_func', 'adam'))
        self.lr_max = params.get('lr_max', 1e-3)
        self.epochs = params.get('epochs', 25)
        self.bs = params.get('batch_size', [64, 128])
        
        if 'loss_func' in params:
            del params['loss_func']
        if 'opt_func' in params:
            del params['opt_func']
        if 'metric_average' in params:
            del params['metric_average']
        if 'batch_tfms' in params:
            del params['batch_tfms']
        if 'lr_max' in params:
            del params['lr_max']
        if 'epochs' in params:
            del params['epochs']
        if 'batch_size' in params:
            del params['batch_size']
        self.params_arch = params
        ############## process params #################
        
    @abstractmethod    
    def _init_learner(self):
        pass
        
    def fit(self):
        self.learn.fit_one_cycle(self.epochs, lr_max=self.lr_max)
        
    @abstractmethod
    def cross_val_score(self, cv=5, scoring='f1'):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def predict_probas(self, X):
        pass
    
    def plot_metrics(self):
        self.learn.plot_metrics()
        
    def plot_probas(self):
        self.learn.show_probas()
        
    def plot_results(self):
        self.learn.show_results()
        
    def plot_confusion_matrix(self):
        interp = ClassificationInterpretation.from_learner(self.learn)
        interp.plot_confusion_matrix()
        
class TsaiModel(FastAIModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average='micro', device='gpu'):
        super(TsaiModel, self).__init__(model_type, params=params, metric_average=metric_average, device=device)
        self.X = X.to_numpy()
        if self.X.ndim == 2:
            self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
        self.y = y.to_numpy()
        
        ############## process params #################
        self.splits = TSSplitter(valid_size=0.2, show_plot=False)
        self.transforms = get_transforms_ts(params.get('batch_tfms', []))
        self.num_classes = y.max() - 1
        if self.num_classes < 6:
            self.metrics = get_metrics_fastai(metric_average, model_type=self.model_type)
        else:
            self.metrics = get_metrics_fastai(metric_average, model_type=self.model_type, add_top_5_acc=True)
        self.splits = TSSplitter(valid_size=0.2, show_plot=False)
        ############## process params #################
            
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
                
    def cross_val_score(self, cv=5, scoring='f1'):
        scores = np.zeros(cv)
        for i in range(cv):
            self._init_learner()
            self.fit()
            if scoring == 'accuracy':
                scores[i] = self.learn.recorder.values[-1][2]
            elif scoring.startswith('f1'):
                scores[i] = self.learn.recorder.values[-1][3]
            elif scoring.startswith('precision'):
                scores[i] = self.learn.recorder.values[-1][4]
            elif scoring.startswith('recall'):
                scores[i] = self.learn.recorder.values[-1][5]
            else:
                logger.warn(f'Unknown scoring: {scoring}. Using f1 instead')
                scores[i] = self.learn.recorder.values[-1][3]
            
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


class PytorchCVModel(FastAIModel): 
        
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict(), metric_average='micro', device='gpu'):
        super(PytorchCVModel, self).__init__(model_type, params=params, metric_average=metric_average, device=device)
        self.params = params
        self.dataset = copy.deepcopy(dataset) # WARNING, maybe problematic for large datasets
        
        ############## process params #################
        self.pretrained = params.get('pretrained', False)
        self.transforms = get_transforms_cv(params.get('batch_tfms', []), params=params.get('batch_tfms_params', dict()))

        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.num_classes = len(dataset.classes)
        else:
            self.num_classes = input('Found no imformation about number of classes. Please enter number of classes: ')
            
        dataset_shape = dataset[0][0].shape
        self.in_channels = dataset_shape[0]
        self.in_size = dataset_shape[1], dataset_shape[2]
        if params.get('batch_tfms', []) != []:
            update_transforms(self.dataset.transform, self.transforms)
        self._check_and_fix_in_size()
        if self.num_classes < 6:
            self.metrics = get_metrics_fastai(metric_average, model_type=self.model_type)
        else:
            self.metrics = get_metrics_fastai(metric_average, model_type=self.model_type, add_top_5_acc=True)
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
            elif scoring.startswith('f1'):
                scores[i] = self.learn.recorder.values[-1][3]
            elif scoring.startswith('precision'):
                scores[i] = self.learn.recorder.values[-1][4]
            elif scoring.startswith('recall'):
                scores[i] = self.learn.recorder.values[-1][5]
            else:
                logger.warn(f'Unknown scoring: {scoring}. Using f1 instead')
                scores[i] = self.learn.recorder.values[-1][3]
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

    def _check_and_fix_in_size(self):    
        if 'resize' not in self.params.get('batch_tfms', []) and self.in_size not in self.std_in_sizes:            
            self.in_size = min(self.std_in_sizes, key=lambda x: abs(x[0]- self.in_size[0]) + abs(x[1]- self.in_size[1]))
            logger.info(f"No resize given. Resize images to standard input size of {self.name}: {self.in_size}")
            new_transforms = [transforms.Resize(self.in_size)]
            update_transforms(self.dataset, new_transforms)
    