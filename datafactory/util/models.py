'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import * 

sys.path.append('../models')
from ..models.model import SklearnModel
from ..models.baseline import BaselineTS, Baseline
from ..models.decision_tree import DecisionTree
from ..models.random_forest import RandomForest
from ..models.ada_boost import AdaBoost
from ..models.knn import KNN
from ..models.svm import SVM
from ..models.gbdt import GBDT
from ..models.gaussian_nb import GaussianNB
from ..models.bayesian_ridge import BayesianRidge
from ..models.inception_time import InceptionTime
from ..models.inception_time_plus import InceptionTimePlus
from ..models.fcn import FCN
from ..models.gru import GRU
from ..models.gru_fcn import GRUFCN
from ..models.lstm import LSTM
from ..models.lstm_fcn import LSTMFCN
from ..models.mlp import MLP
from ..models.mwdn import MWDN
from ..models.omni_scale import OmniScale
from ..models.res_cnn import ResCNN
from ..models.res_net import ResNet
from ..models.tab_model import TabModel
from ..models.tcn import TCN
from ..models.tst import TST
from ..models.xception_time import XceptionTime
from ..models.xcm import XCM

from .constants import logger

def _get_model(model, X, y, model_type, average=None):
    if model == 'baseline_ts':
        return BaselineTS(X, y, model_type)
    elif model == 'baseline':
        return Baseline(X, y, model_type)
    elif model == 'decision_tree':
        return DecisionTree(X, y, model_type)
    elif model == 'random_forest':  
        return RandomForest(X, y, model_type)
    elif model == 'ada_boost':
        return AdaBoost(X, y, model_type)
    elif model == 'svm':
        return SVM(X, y, model_type)
    elif model == 'knn':
        return KNN(X, y, model_type)
    elif model == 'gbdt':
        return GBDT(X, y, model_type)
    elif model == 'gaussian_nb':
        return GaussianNB(X, y, model_type)
    elif model == 'bayesian_ridge':
        return BayesianRidge(X, y, model_type)
    elif model == 'inception_time':
        return InceptionTime(X, y, model_type, metric_average=average)
    elif model == 'inception_time_plus':
        return InceptionTimePlus(X, y, model_type, metric_average=average)
    elif model == 'fcn':
        return FCN(X, y, model_type, metric_average=average)
    elif model == 'gru':
        return GRU(X, y, model_type, metric_average=average)
    elif model == 'gru_fcn':
        return GRUFCN(X, y, model_type, metric_average=average)
    elif model == 'lstm':
        return LSTM(X, y, model_type, metric_average=average)
    elif model == 'lstm_fcn':
        return LSTMFCN(X, y, model_type, metric_average=average)
    elif model == 'mlp':
        return MLP(X, y, model_type, metric_average=average)
    elif model == 'mwdn':
        return MWDN(X, y, model_type, metric_average=average)
    elif model == 'omni_scale':
        return OmniScale(X, y, model_type, metric_average=average)
    elif model == 'res_cnn':
        return ResCNN(X, y, model_type, metric_average=average)
    elif model == 'res_net':
        return ResNet(X, y, model_type, metric_average=average)
    elif model == 'tab_model':
        return TabModel(X, y, model_type, metric_average=average)
    elif model == 'tcn':
        return TCN(X, y, model_type, metric_average=average)
    elif model == 'tst':
        return TST(X, y, model_type, metric_average=average)
    elif model == 'xception_time':
        return XceptionTime(X, y, model_type, metric_average=average)
    elif model == 'xcm':
        return XCM(X, y, model_type, metric_average=average)
    else:
        logger.warn(f'Skipping model. Unknown model: {model}')
        
def get_available_models_and_metrics(model_type):
    available_models_classification = [{'label': 'Baseline', 'value': 'baseline_ts'},
                                       {'label': 'Decision Tree', 'value': 'decision_tree'},
                                       {'label': 'RandomF orest', 'value': 'random_forest'},
                                       {'label': 'AdaBoost', 'value': 'ada_boost'},
                                       {'label': 'SVM', 'value': 'svm'},
                                       {'label': 'KNN', 'value': 'knn'},
                                       {'label': 'GBDT', 'value': 'gbdt'},
                                       {'label': 'Gaussian NB', 'value': 'gaussian_nb'},
                                       {'label': 'InceptionTime', 'value': 'inception_time'},
                                       {'label': 'InceptionTimePlus', 'value': 'inception_time_plus'},
                                       {'label': 'FCN', 'value': 'fcn'},
                                       {'label': 'GRU', 'value': 'gru'},
                                       {'label': 'GRUFCN', 'value': 'gru_fcn'},
                                       {'label': 'LSTM', 'value': 'lstm'},
                                       {'label': 'LSTMFCN', 'value': 'lstm_fcn'},
                                       {'label': 'MLP', 'value': 'mlp'},
                                       {'label': 'MWDN', 'value': 'mwdn'},
                                       {'label': 'OmniScale', 'value': 'omni_scale'},
                                       {'label': 'ResCNN', 'value': 'res_cnn'},
                                       {'label': 'ResNet', 'value': 'res_net'},
                                       {'label': 'TabModel', 'value': 'tab_model'},
                                       {'label': 'tcn', 'value': 'TCN'},
                                       {'label': 'TST', 'value': 'tst'},
                                       {'label': 'XceptionTime', 'value': 'xception_time'},
                                       {'label': 'XCM', 'value': 'xcm'}]
    
    available_models_regression = [{'label': 'Baseline', 'value': 'baseline_ts'},
                                       {'label': 'Decision Tree', 'value': 'decision_tree'},
                                       {'label': 'Random Forest', 'value': 'random_forest'},
                                       {'label': 'AdaBoost', 'value': 'ada_boost'},
                                       {'label': 'SVM', 'value': 'svm'},
                                       {'label': 'KNN', 'value': 'knn'},
                                       {'label': 'GBDT', 'value': 'gbdt'},
                                       {'label': 'Bayesian Ridge', 'value': 'bayesian_ridge'},
                                       {'label': 'InceptionTime', 'value': 'inception_time'},
                                       {'label': 'InceptionTimePlus', 'value': 'inception_time_plus'},
                                       {'label': 'FCN', 'value': 'fcn'},
                                       {'label': 'GRU', 'value': 'gru'},
                                       {'label': 'GRUFCN', 'value': 'gru_fcn'},
                                       {'label': 'LSTM', 'value': 'lstm'},
                                       {'label': 'LSTMFCN', 'value': 'lstm_fcn'},
                                       {'label': 'MLP', 'value': 'mlp'},
                                       {'label': 'MWDN', 'value': 'mwdn'},
                                       {'label': 'OmniScale', 'value': 'omni_scale'},
                                       {'label': 'ResCNN', 'value': 'res_cnn'},
                                       {'label': 'ResNet', 'value': 'res_net'},
                                       {'label': 'TabModel', 'value': 'tab_model'},
                                       {'label': 'TCN', 'value': 'tcn'},
                                       {'label': 'TST', 'value': 'tst'},
                                       {'label': 'XceptionTime', 'value': 'xception_time'},
                                       {'label': 'XCM', 'value': 'xcm'}]
    
    average = ['binary', 'weighted', 'micro', 'macro', 'samples']
    
    if model_type == 'C':
        available_models = available_models_classification
        scoring = ['accuracy', 'precision', 'f1', 'recall']
    elif model_type == 'R':
        available_models = available_models_regression
        scoring = ['explained_variance','r2', 'mae', 'mse']
    else:
        available_models = available_models_regression
        
    return available_models, scoring, average
        
def get_models_from_list(model_type:str, models_values: List=None):
    available_models, _, _ = get_available_models_and_metrics(model_type)
    models = []
    if not models is None:
        for m in available_models:
            if m['value'] in models_values:
                models.append(m)  
    return models

def get_labels_from_values(model_type, models_values):
    available_models, _, _ = get_available_models_and_metrics(model_type)
    labels = []
    for i in available_models:
        if i['value'] in models_values:
            labels.append(i['label'])
        
    return labels
        