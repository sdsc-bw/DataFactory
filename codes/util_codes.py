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
    
def get_loss(loss:str):
    # TODO add more
    if loss == 'cross_entropy':
        return CrossEntropyLossFlat()
    elif loss == 'mse':
        return MSELossFlat()
    elif loss == 'smooth_cross_entropy':
        return LabelSmoothingCrossEntropyFlat()
    else:
        return None
        
def get_optimizer(optimizer:str):
    # TODO add more
    if optimizer == 'adam':
        return Adam
    else:
        return Adam
        
def get_metrics(metrics: list):
    metrics_list = []
    metrics_list.append(accuracy)
    # TODO add more
    #if 'accuracy' in metrics:
        #metrics_list.append(accuracy)
            
    return metrics_list

def get_transforms(transforms: list):
    transforms_list = []
    if 'standardize' in transforms:
        transforms_list.append(TSStandardize())
    # TODO add more
    return transforms_list

def contains_tsai_model(models):
    # TODO add more
    if 'inception_time' in models:
        return True

def get_library(model):
    if 'inception_time' == model:
        return 'tsai'
    else:
        return 'sklearn'