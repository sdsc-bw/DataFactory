from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split, cross_val_score
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import * # TODO move to init

class Model(metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str):
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
    def predict(self, y: pd.Series):
        pass
    
    @abstractmethod
    def predict_probas(self, y: pd.Series):
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
        print(self.name)
        score = cross_val_score(self.model, self.X, self.y, cv=cv, scoring=scoring)
        return score
    
    def predict(self, y: pd.Series):
        return self.model.predict(y)
    
    def predict_probas(self, y: pd.Series):
        return self.model.predict_probas(y)
    
class TsaiModel(Model):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(TsaiModel, self).__init__(X, y, mtype)
        self.params = params
        
        ############## process params #################
        params = self.params.copy()
        self.loss = self.get_loss(params.get('loss_func', 'cross_entropy'))
        self.optim = self.get_optimizer(params.get('opt_func', 'adam'))
        self.metrics = self.get_metrics(params.get('metrics', 'accuracy')) 
        self.transforms = self.get_transforms(params.get('batch_tfms', []))
        self.lr_max = params.get('lr_max', 1e-3)
        self.epochs = params.get('epochs', 1e-3)
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
    
    def predict(self, y: pd.Series):
        # TODO
        pass
    
    def predict_probas(self, y: pd.Series):
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
        elif loss == 'mse':
            return MSELossFlat() # Regression/Forecasting
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

    