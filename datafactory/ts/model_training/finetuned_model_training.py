'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''
import pandas as pd
import hyperopt
import time
import sys
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
import mlflow
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import *#
from statistics import mean
from IPython.display import clear_output

from .search_space import std_search_space

sys.path.append('../../models')
from ...models.model import SklearnModel
from ...models.decision_tree import DecisionTree
from ...models.random_forest import RandomForest
from ...models.ada_boost import AdaBoost
from ...models.knn import KNN
from ...models.svm import SVM
from ...models.gbdt import GBDT
from ...models.gaussian_nb import GaussianNB
from ...models.bayesian_ridge import BayesianRidge
from ...models.inception_time import InceptionTime
from ...models.inception_time_plus import InceptionTimePlus
from ...models.fcn import FCN
from ...models.gru import GRU
from ...models.gru_fcn import GRUFCN
from ...models.lstm import LSTM
from ...models.lstm_fcn import LSTMFCN
from ...models.mlp import MLP
from ...models.mwdn import MWDN
from ...models.omni_scale import OmniScale
from ...models.res_cnn import ResCNN
from ...models.res_net import ResNet
from ...models.tab_model import TabModel
from ...models.tcn import TCN
from ...models.tst import TST
from ...models.xception_time import XceptionTime
from ...models.xcm import XCM

sys.path.append('../../util')
from ...util.constants import logger

TEMP_X = None
TEMP_Y = None
MODEL_TYPE = None
CV = 5
RESULTS = None
COUNTER = 0
SCORING = 'f1'
AVERAGE = 'micro'

def finetune(X: pd.DataFrame, y: pd.Series, strategy: str='random', models: list=['decision_tree'], params: Dict=dict(), max_evals: int=32, cv: int=5, model_type: str='C', scoring='f1', average='micro'):
    """Finetunes one or multiple models according with hyperopt.
        
    Keyword arguments:
    X -- data
    y -- targets
    strategy -- strategy to select models for finetuning should be in ['random', 'parzen']
    models -- models should be from list of avaiable models
    params -- list of dictionaries with parameter to try out
    max_evals -- maximal number of models to finetune
    cv -- number of trainings during cross validation
    model_type -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
    Output:
    the model with the highest score
    """
    global TEMP_X
    global TEMP_Y
    global MODEL_TYPE
    global CV 
    global RESULTS
    global SCORING
    global AVERAGE
    global COUNTER
    TEMP_X = X
    TEMP_Y = y
    MODEL_TYPE = model_type
    CV = cv
    AVERAGE = average  
    
    if type(scoring) != list:
        scoring = [scoring]
    
    # concat scoring and average
    for i in range(len(scoring)):
        if scoring[i] != 'accuracy' and scoring[i] != 'mse' and scoring[i] != 'mae' and scoring[i] != 'explained_variance' and scoring[i] != 'r2':
            scoring[i] = scoring[i] + '_' + average      
        
    SCORING = scoring    
    RESULTS = pd.DataFrame(columns=['model'] + list(map(lambda x: 'test_' + x, scoring)))
        
    trials = Trials()
    
    search_space = _get_search_spaces(models, params)
    
    if strategy == 'parzen':
        algo = tpe.suggest
    elif strategy == 'random':
        algo = rand.suggest
    else:
        raise ValueError(f'Unknown strategy: {strategy}')
    
    with mlflow.start_run():
        best_result = fmin(
        fn=_objective, 
        space=search_space,
        algo=algo,
        max_evals=max_evals,
        trials=trials)
        
    #best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
    
    TEMP_X = None
    TEMP_Y = None
    MODEL_TYPE = None
    CV = 5
    SCORING = 'f1'
    AVERAGE = 'micro'
    COUNTER = 0
    
    return RESULTS

def _objective(params):
    global CV
    global RESULTS
    global SCORING
    global COUNTER
    start = time.time()
    model = params['model']
    del params['model']
        
    model = _get_model(model, params)
        
    clear_output()
    display(RESULTS)
    
    logger.info("Running cross validation for: " + model.get_name() + "...")
    score = model.cross_val_score(cv=CV, scoring=SCORING)
            
    
    clear_output()

    for i in range(CV):
        
        params = model.get_params()
        model_name = f'{model.get_name()}('
        for key in params:
            model_name += f'{key}={params[key]},'
        model_name += ')'
        
        RESULTS.loc[COUNTER, 'model'] = model_name
        for j in SCORING:
            RESULTS.loc[COUNTER, 'test_' + j] = score[j][i]
        
        COUNTER += 1
    
    display(RESULTS)
    
    # clear cache
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'loss': mean(score[SCORING[0]]), 'status': STATUS_OK}

def _get_model(model, params=dict()):
    global TEMP_X
    global TEMP_Y
    global MODEL_TYPE
    global AVERAGE
    if model == 'decision_tree':
        return DecisionTree(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'random_forest':  
        return RandomForest(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'ada_boost':
        return AdaBoost(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'svm':
        return SVM(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'knn':
        return KNN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'gbdt':
        return GBDT(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'gaussian_nb':
        return GaussianNB(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'bayesian_ridge':
        return BayesianRidge(TEMP_X, TEMP_Y, MODEL_TYPE, params=params)
    elif model == 'inception_time':
        return InceptionTime(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'inception_time_plus':
        return InceptionTimePlus(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'fcn':
        return FCN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'gru':
        return GRU(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'gru_fcn':
        return GRUFCN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'lstm':
        return LSTM(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'lstm_fcn':
        return LSTMFCN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'mlp':
        return MLP(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'mwdn':
        return MWDN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'omni_scale':
        return OmniScale(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'res_cnn':
        return ResCNN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'res_net':
        return ResNet(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'tab_model':
        return TabModel(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'tcn':
        return TCN(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'tst':
        return TST(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'xception_time':
        return XceptionTime(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'xcm':
        return XCM(TEMP_X, TEMP_Y, MODEL_TYPE, params=params, metric_average=AVERAGE)
    else:
        logger.warn(f'Skipping model. Unknown model: {model}')

def _get_search_spaces(models: list, params):
    global MODEL_TYPE
    search_space_list = []
    for model in models:
        if model not in params:
            if model == 'decision_tree':
                if MODEL_TYPE == 'C':
                    model_space = std_search_space[model + '_c'].copy()
                else:
                    model_space = std_search_space[model + '_r'].copy()
            elif model in std_search_space:
                model_space = std_search_space[model].copy()
            else:
                raise ValueError(f'Unknown model: {model}')
        else:
            model_space = params[model]
            model_space['model'] = model
        search_space_list.append(model_space)    

    search_space_model = hp.choice('classifier_type', search_space_list)
    return search_space_model