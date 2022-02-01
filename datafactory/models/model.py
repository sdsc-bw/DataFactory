'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import *
import torch
import torch.optim
from torch.utils.data import DataLoader, Dataset

class Model(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, X, y, mtype: str):
        self.X = X
        self.y = y
        self.mtype = mtype
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
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(SklearnModel, self).__init__(X, y, mtype)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.params = params
        self.model = None
        
    def fit(self):
        model.fit(self.X_train, self.y_test)
        
#    def evaluate(self, metric='accuracy'):

    def cross_val_score(self, cv=5, scoring='accuracy'):
        score = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring)
        return score
    
    def predict(self, X: pd.Series):
        return self.model.predict(X)
    
    def predict_probas(self, X: pd.Series):
        return self.model.predict_probas(X)
    
class TsaiModel(Model):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(TsaiModel, self).__init__(X, y, mtype)
        self.params = params
        
        ############## process params #################
        params = self.params.copy()
        self.loss = self.get_loss(params.get('loss_func', 'cross_entropy'))
        self.optimizer = self.get_optimizer(params.get('opt_func', 'adam'))
        self.metrics = self.get_metrics(params.get('metrics', ['accuracy'])) 
        self.transforms = self.get_transforms(params.get('batch_tfms', []))
        self.lr_max = params.get('lr_max', 1e-3)
        self.epochs = params.get('epochs', 25)
        self.splits = params.get('splits', None)
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
        if self.mtype == 'C':
            learner = TSClassifier
        elif self.mtype == 'R':
            learner = TSRegressor
        elif self.mtype == 'F':
            learner = TSForecaster
        else:
            logger.error('Unknown type of model')
        self.learn = learner(self.X, self.y, bs=self.bs, batch_tfms=self.transforms, arch=self.arch, 
                             metrics=self.metrics, arch_config=self.params_arch)
       
    def fit(self):
        self.learn.fit_one_cycle(self.epochs, lr_max=self.lr_max)
        
    def cross_val_score(self, cv=5, scoring='accuracy'):
        if scoring == 'accuracy':
            scores = np.zeros(cv)
            for i in range(cv):
                self._init_learner()
                self.fit()
                ## TODO change metric from accuracy to f1
                scores[i] = self.learn.recorder.values[-1][2]
        clear_output()
        return scores
    
    def predict(self, X: pd.Series):
        # TODO
        pass
    
    def predict_probas(self, X: pd.Series):
        # TODO
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
    
    def get_params(self):
        params = self.params.copy()
        if 'metrics' in params:
            del params['metrics']
        return params
    
    @staticmethod
    def get_loss(loss:str):
        if loss == 'cross_entropy':
            return CrossEntropyLossFlat() # Classification
        elif loss == 'smooth_cross_entropy':
            return LabelSmoothingCrossEntropyFlat() # Classification
        elif loss == 'l1':
            return L1LossFlat() # Regression/Forecasting
        elif loss == 'focal':
            return FocalLoss() # Classification
        elif loss == 'dice':
            return DiceLoss() # Classification
        elif loss == 'bce':
            return BCEWithLogitsLossFlat() # Regression/Forecasting
        else:
            return None
                             
    @staticmethod                        
    def get_optimizer(optimizer:str):
        if optimizer == 'adam':
            return Adam
        if optimizer == 'r_adam':
            return RAdam
        if optimizer == 'qh_adam':
            return QHAdam
        if optimizer == 'sgd':
            return SGD
        if optimizer == 'rms_prop':
            return RMSProp    
        if optimizer == 'larc':
            return Larc
        if optimizer == 'lamb':
            return Lamb
        else:
            return Adam
     
    @staticmethod
    def get_metrics(metrics: list):
        metrics_list = []
        metrics_list.append(accuracy)
        if 'mae' in metrics:
            metrics_list.append(mae)
        if 'mse' in metrics:
            metrics_list.append(mse)
        if 'top_k_accuracy' in metrics:
            metrics_list.append(top_k_accuracy)   
        return metrics_list

    @staticmethod
    def get_transforms(transforms: list):
        transforms_list = []
        if 'standardize' in transforms:
            transforms_list.append(TSStandardize())
        if 'clip' in transforms:
            transforms_list.append(TSClip())    
        if 'mag_scale' in transforms:
            transforms_list.append(TSMagScale())
        if 'window_wrap' in transforms:
            transforms_list.append(TSWindowWarp())    
        return transforms_list

class PytorchCVModel(Model): 
        
    def __init__(self, dataset: Dataset, mtype: str, params:Dict=dict()):
        self.params = params
        self.dataset = dataset
        
        ############## process params #################
        params = self.params.copy()
        self.lr_max = params.get('lr_max', 1e-3)
        self.epochs = params.get('epochs', 25)
        self.bs = params.get('batch_size', 64)
        self._init_training_params()
        ############## process params #################
        
        self.train_size = int(0.8 * len(dataset))
        self.test_size = len(dataset) - self.train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)  
        self.test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)
        self.model = self._init_model()
     
    @abstractmethod
    def _init_model(self):
        pass

    def _init_training_params(self):
        self.loss = self.get_loss(self.params.get('loss_func', 'cross_entropy'))
        self.optimizer = self.get_optimizer(self.params.get('opt_func', 'adam'))  
        
    def fit(self):
        self._train(self.train_loader)
                
    def evaluate(self):
        self._test(self.test_loader)

    def cross_val_score(self, cv=5, scoring='accuracy'):
        if scoring == 'accuracy':
            scores = np.zeros(cv)
            for i in range(cv):
                train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, 
                                                                            [self.train_size, self.test_size], 
                                                                            generator=torch.Generator().manual_seed(i))
                train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)  
                test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)
                self._init_model()
                self._init_training_params()
                self._train(train_loader)
                ## TODO change metric from accuracy to f1
                scores[i] = self._test(test_loader)
        #clear_output()
        return scores
    

    def predict(self, X):
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().detach().numpy()
    
    def predict_probas(self, X):
        outputs = self.model(images)
        return outputs.cpu().detach().numpy()
    
    def _train(self, train_loader):
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
    def _test(self, test_loader):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        ## TODO change metric from accuracy to f1    
        results = correct / total
        return results
    
    @staticmethod
    def get_loss(loss:str):
        if loss == 'cross_entropy':
            return torch.nn.CrossEntropyLoss() # Classification
        elif loss == 'nll':
            return torch.nn.NLLLoss()
        elif loss == 'hinge':
            return torch.nn.HingeEmbeddingLoss()
        elif loss == 'kl_div':
            return torch.nn.KLDivLoss()
        else:
            return torch.nn.CrossEntropyLoss()
                                                   
    def get_optimizer(self, optimizer:str):
        if optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr_max)
        if optimizer == 'r_adam':
            return torch.optim.RAdam(self.model.parameters(), lr=self.lr_max)
        if optimizer == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=self.lr_max)
        if optimizer == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.lr_max)
        if optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=self.lr_max)
        if optimizer == 'rms_prop':
            return optim.RMSprop(model.parameters(), lr=self.lr_max)    
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.lr_max)
    