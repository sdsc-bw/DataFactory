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
from torchvision import transforms
import gc
import time
import copy

sys.path.append('../util')
from ..util.metrics import val_score
from ..util.transforms import get_transforms_cv

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
        self.loss = self.get_loss(params.get('loss_func', 'cross_entropy'))
        self.optimizer = self.get_optimizer(params.get('opt_func', 'adam'))
        self.metrics = self.get_metrics(params.get('metrics', ['accuracy'])) 
        self.transforms = self.get_transforms(params.get('batch_tfms', []))
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
        if scoring == 'accuracy':
            scores = np.zeros(cv)
            for i in range(cv):
                self._init_learner()
                self.fit()
                scores[i] = self.learn.recorder.values[-1][2]
        else:
            scores = np.zeros(cv)
            for i in range(cv):
                self._init_learner()
                self.fit()
                _, targets, preds = self.learn.get_X_preds(self.shuffled_X[self.splits[1]], self.shuffled_y[self.splits[1]])
                targets = targets.cpu().detach().numpy().tolist()
                preds = [int(p) for p in preds]
                scores[i] = val_score(targets, preds, scoring)
        clear_output()
        return scores
    
    def predict(self, X: pd.Series):
        if X.ndim == 2:
            X = self.X.reshape(X.shape[0], X.shape[1], 1)
        _, _, preds = learn.get_X_preds(X)
        return preds
    
    def predict_probas(self, X: pd.Series):
        if X.ndim == 2:
            X = self.X.reshape(X.shape[0], X.shape[1], 1)
        probas, _, _ = learn.get_X_preds(X)
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
            logger.warn(f'Unknown loss: {loss}. Using default loss instead')
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
            logger.warn(f'Unknown optimizer: {optimizer}. Using Adam instead')
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
        
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict()):
        super(PytorchCVModel, self).__init__(model_type)
        self.params = params
        self.dataset = copy.deepcopy(dataset) # WARNING, maybe problematic for large datasets
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        ############## process params #################
        params = self.params.copy()
        self.lr_max = params.get('lr_max', 1e-3)
        self.epochs = params.get('epochs', 100)
        self.bs = params.get('batch_size', 16) 
        self.pretrained = params.get('pretrained', False)
        print(self.params)
        print(params.get('batch_tfms', []))
        self.transforms = get_transforms_cv(params.get('batch_tfms', []), params=params.get('batch_tfms_params', dict()))
        self.num_classes = len(dataset.classes)
        dataset_shape = dataset[0][0].shape
        self.in_channels = dataset_shape[0]
        self.in_size = dataset_shape[1], dataset_shape[2]
        ############## process params #################
        
        self.train_size = int(0.8 * len(dataset))
        self.test_size = len(dataset) - self.train_size
        #self._update_transforms(self.transforms.transforms)
        self._check_and_fix_in_size()
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)  
        self.test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)
     
    @abstractmethod
    def _init_model(self):
        pass

    def _init_training_params(self):
        self.loss = self.get_loss(self.params.get('loss_func', 'cross_entropy'))
        self.optimizer = self.get_optimizer(self.params.get('opt_func', 'adam'))  
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        torch.cuda.empty_cache()
        gc.collect()
        
    def fit(self):
        self._train(self.train_loader)
                
    def evaluate(self):
        self._test(self.test_loader)

    def cross_val_score(self, cv: int=5, scoring: str='f1'):
        scores = np.zeros(cv)
        test_losses = np.zeros(cv)
        for i in range(cv):
            train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.test_size], 
                                                                        generator=torch.Generator().manual_seed(i))
            train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)  
            test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False)
            self._init_model()
            self._init_training_params()
            self._train(train_loader)
            test_losses[i], scores[i] = self._test(test_loader, scoring=scoring)

        #clear_output()
        return scores    

    def predict(self, X):
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().detach().numpy()
    
    def predict_probas(self, X):
        outputs = self.model(images)
        return outputs.cpu().detach().numpy()
    
    def _train_and_test(self, train_loader, test_loader):       
        self._init_model()
        self._init_training_params()
        results = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'time'])
        for epoch in range(self.epochs):
            start = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            test_loss, test_acc = self._test(test_loader)
            elapsed = time.time() - start
            results.loc[i] = [epoch, train_loss, test_loss, int(elapsed)]
            #clear_output()
            display(results)
    
    def _train(self, train_loader):
        self._init_model()
        self._init_training_params()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.params.get('loss_func', 'cross_entropy') == 'nll':
                    m = nn.LogSoftmax(dim=1)
                    loss = self.loss(m(outputs), labels)
                else:
                    loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()    

    def _test(self, test_loader: DataLoader, scoring: str='f1'):
        correct = 0
        total = 0
        running_loss = 0.0
        preds = []
        targets = []
        self.model.eval()
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if self.params.get('loss_func', 'cross_entropy') == 'nll':
                    m = nn.LogSoftmax(dim=1)
                    loss = self.loss(m(outputs), labels)
                else:
                    loss = self.loss(outputs, labels)
                running_loss += loss.item()
                
                for p in predicted:
                    preds.append(p.cpu().detach().numpy().tolist())
                for t in labels:
                    targets.append(t.cpu().detach().numpy().tolist())  
        if scoring == 'accuracy':
            score = correct / total
        else:
            score = val_score(targets, preds, scoring)
        return running_loss, score            
            
    def _train_epoch(self, train_loader):
        correct = 0
        total = 0
        running_loss = 0.0
        self.model.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if self.params.get('loss_func', 'cross_entropy') == 'nll':
                m = nn.LogSoftmax(dim=1)
                loss = self.loss(m(outputs), labels)
            else:
                loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        self.scheduler.step()
        return running_loss, accuracy                            
    
    def _check_and_fix_in_size(self):
        if self.in_size not in self.available_in_sizes:
            self.in_size = min(self.available_in_sizes, key=lambda x: abs(x[0]- self.in_size[0]) + abs(x[1]- self.in_size[1]))
        new_transforms = [transforms.Resize(self.in_size)]
        self._update_transforms(new_transforms)
    
    def _update_transforms(self, new_transforms):
        curr_transforms = self.dataset.transform.transforms
        new_transforms = curr_transforms + new_transforms
        self.dataset.transform = transforms.Compose(new_transforms)
    
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
            raise ValueError(f'Unknown loss: {loss}')
                                                   
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
            logger.warn(f'Unknown optimizer: {optimizer}. Using Adam instead')
            return torch.optim.Adam(self.model.parameters(), lr=self.lr_max)
    