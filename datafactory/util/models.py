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