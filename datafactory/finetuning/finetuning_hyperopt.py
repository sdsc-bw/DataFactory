import pandas as pd
import hyperopt
import time
import sys
from hyperopt import fmin, tpe, rand, hp, SparkTrials, STATUS_OK, Trials
import mlflow
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import *

from .search_space_hyperopt import std_search_space

sys.path.append('../models')
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
from ..models.mini_rocket import MiniRocket
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

sys.path.append('../util')
from ..util.constants import logger

def finetune_hyperopt(X: pd.DataFrame, y: pd.Series, strategy: str='random', models: list=['decision_tree'], params: Dict=dict(), max_evals: int=32, cv: int=5, mtype: str='C'):
    """Finetunes one or multiple models according with hyperopt.
        
    Keyword arguments:
    X -- data
    y -- targets
    strategy -- strategy to select models for finetuning should be in ['random', 'parzen']
    models -- models should be from list of avaiable models
    params -- list of dictionaries with parameter to try out
    max_evals -- maximal number of models to finetune
    cv -- number of trainings during cross validation
    mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
    Output:
    the model with the highest score
    """
    trials = Trials()
            
    search_space = _get_search_spaces(X, y, models, mtype, cv, params)
    
    if strategy == 'parzen':
        algo = tpe.suggest
    elif strategy == 'random':
        algo = rand.suggest
    else:
        logger.error('Unknown strategy')
    
    with mlflow.start_run():
        best_result = fmin(
        fn=_objective, 
        space=search_space,
        algo=algo,
        max_evals=max_evals,
        trials=trials)
        
    best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        
    return best_model

def _objective(params):
    start = time.time()
    model = params['model']
    cv = params['cv']
    results = params['results']
    del params['model']
    del params['cv']
    del params['results']
        
    model = _get_model(model, params)
        
    clear_output()
    display(results)
        
    # TODO use f1 instead of accuracy
    
    logger.info("Running cross validation for: " + model.get_name() + "...")
    score = model.cross_val_score(cv=cv, scoring='accuracy').mean() 
            
    elapsed = time.time() - start
    
    clear_output()
    results.loc[len(results)] = [model.get_name(), score, model.get_params(), elapsed]
    results.sort_values(by='Score', ascending=False, ignore_index=True, inplace=True)
    display(results)
    
    return {'loss': -score, 'status': STATUS_OK, 'model': model, 'elapsed': elapsed}

def _get_model(model, params=dict()):
    X = params['X']
    y = params['y']
    mtype = params['mtype']
    del params['X']
    del params['y']
    del params['mtype']
    if model == 'decision_tree':
        return DecisionTree(X, y, mtype, params=params)
    elif model == 'random_forest':  
        return RandomForest(X, y, mtype, params=params)
    elif model == 'ada_boost':
        return AdaBoost(X, y, mtype, params=params)
    if model == 'svm':
        return SVM(X, y, mtype, params=params)
    if model == 'knn':
        return KNN(X, y, mtype, params=params)
    if model == 'gbdt':
        return GBDT(X, y, mtype, params=params)
    if model == 'gaussian_nb':
        return GaussianNB(X, y, mtype, params=params)
    if model == 'bayesian_ridge':
        return BayesianRidge(X, y, mtype, params=params)
    elif model == 'inception_time':
        return InceptionTime(X, y, mtype, params=params)
    elif model == 'inception_time_plus':
        return InceptionTimePlus(X, y, mtype, params=params)
    elif model == 'fcn':
        return FCN(X, y, mtype, params=params)
    elif model == 'gru':
        return GRU(X, y, mtype, params=params)
    elif model == 'gru_fcn':
        return GRUFCN(X, y, mtype, params=params)
    elif model == 'lstm':
        return LSTM(X, y, mtype, params=params)
    elif model == 'lstm_fcn':
        return LSTMFCN(X, y, mtype, params=params)
    elif model == 'mini_rocket':
        return MiniRocket(X, y, mtype, params=params)
    elif model == 'mlp':
        return MLP(X, y, mtype, params=params)
    elif model == 'mwdn':
        return MWDN(X, y, mtype, params=params)
    elif model == 'omni_scale':
        return OmniScale(X, y, mtype, params=params)
    elif model == 'res_cnn':
        return ResCNN(X, y, mtype, params=params)
    elif model == 'res_net':
        return ResNet(X, y, mtype, params=params)
    elif model == 'tab_model':
        return TabModel(X, y, mtype, params=params)
    elif model == 'tcn':
        return TCN(X, y, mtype, params=params)
    elif model == 'tst':
        return TST(X, y, mtype, params=params)
    elif model == 'xception_time':
        return XceptionTime(X, y, mtype, params=params)
    elif model == 'xcm':
        return XCM(X, y, mtype, params=params)
    else:
        logger.error('Unknown model: ' + model)

def _get_search_spaces(X: pd.DataFrame, y: pd.Series, models: list, mtype:str, cv:int, params):
    search_space_list = []
    results = pd.DataFrame(columns=['Model', 'Score', 'Hyperparams', 'Time'])
    const_params = {'X': X, 'y': y, 'mtype': mtype, 'cv': cv, 'results': results}
    for model in models:
        if model not in params:
            if model == 'decision_tree':
                if mtype == 'C':
                    model_space = std_search_space[model + '_c'].copy()
                else:
                    model_space = std_search_space[model + '_r'].copy()
            else:
                model_space = std_search_space[model].copy()
        else:
            model_space = params[model]
            model_space['model'] = model
        model_space.update(const_params)
        search_space_list.append(model_space)    

    search_space_model = hp.choice('classifier_type', search_space_list)
    return search_space_model