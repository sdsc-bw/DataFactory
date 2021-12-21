import sklearn
import sys
from tsai.all import *
computer_setup()

def model_to_string(model):
    if type(model) == sklearn.tree._classes.DecisionTreeClassifier or type(model) == sklearn.tree._classes.DecisionTreeRegressor:
        return 'decision_tree'
    elif  type(model) == sklearn.ensemble._forest.RandomForestClassifier or type(model) == sklearn.ensemble._forest.RandomForestRegressor:
        return 'random_forest'
    elif type(model) == sklearn.ensemble._weight_boosting.AdaBoostClassifier or type(model) == sklearn.ensemble._weight_boosting.AdaBoostRegressor:
        return 'adaboost'
    elif type(model) == sklearn.neighbors._classification.KNeighborsClassifier or type(model) == sklearn.neighbors._regression.KNeighborsRegressor:
        return 'knn'
    elif type(model) == sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier or type(model) == sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor:
        return 'gbdt'
    elif type(model) == sklearn.naive_bayes.GaussianNB:
        return 'gaussian_nb'
    elif  type(model) == sklearn.svm._classes.SVC == sklearn.svm._classes.SVR:
        return 'svm'
    elif type(model) == sklearn.linear_model._bayes.BayesianRidge:
        return 'bayesian'
    elif model.model.__name__ == 'InceptionTime':
        return 'inception_time'
    elif model.model.__name__ == 'InceptionTimePlus':
        return 'inception_time_plus'
    elif model.model.__name__ == 'LSTM':
        return 'lstm'
    elif model.model.__name__ == 'GRU':
        return 'gru'
    elif model.model.__name__ == 'MLP':
        return 'mlp'
    elif model.model.__name__ == 'FCN':
        return 'fcn'
    elif model.model.__name__ == 'ResNet':
        return 'res_net'
    elif model.model.__name__ == 'LSTM-FCN':
        return 'lstm_fcn'
    elif model.model.__name__ == 'LSTM-GRU':
        return 'lstm_gru'
    elif model.model.__name__ == 'mWDN':
        return 'mwdn'
    elif model.model.__name__ == 'TCN':
        return 'tcn'
    elif model.model.__name__ == 'XceptionTime':
        return 'xception_time'
    elif model.model.__name__ == 'ResCNN':
        return 'res_cnn'
    elif model.model.__name__ == 'TabModel':
        return 'tab_model'
    elif model.model.__name__ == 'OmniScale':
        return 'omni_scale'
    elif model.model.__name__ == 'TST':
        return 'tst'
    elif model.model.__name__ == 'XCM':
        return 'xcm'
    elif model.model.__name__ == 'MiniRocket':
        return 'mini_rocket'
    
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

def get_library(model):
    if 'inception_time' == model:
        return 'tsai'
    if 'inception_time_plus' == model:
        return 'tsai'
    elif model == 'lstm':
        return 'tsai'
    elif model == 'gru':
        return 'tsai'
    elif model == 'mlp':
        return 'tsai'
    elif model == 'fcn':
        return 'tsai'
    elif model == 'res_net':
        return 'tsai'
    elif model == 'lstm_fcn':
        return 'tsai'
    elif model == 'gru_fcn':
        return 'tsai'
    elif model == 'mwdn':
        return 'tsai'
    elif model == 'tcn':
        return 'tsai'
    elif model == 'xception_time':
        return 'tsai'
    elif model == 'res_cnn':
        return 'tsai'
    elif model == 'tab_model':
        return 'tsai'
    elif model == 'omni_scale':
        return 'tsai'
    elif model == 'tst':
        return 'tsai'
    elif model == 'xcm':
        return 'tsai'
    elif model == 'mini_rocket':
        return 'tsai'
    else:
        return 'sklearn'