'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
import hyperopt
from IPython.display import clear_output
import time
import sys
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
import mlflow
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import gc

from .search_space import std_search_space

sys.path.append('../../models')
from ...models import SKLEARN_MODELS
from ...models.model import SklearnModel
from ...models.decision_tree import DecisionTree
from ...models.random_forest import RandomForest
from ...models.ada_boost import AdaBoost
from ...models.knn import KNN
from ...models.svm import SVM
from ...models.gbdt import GBDT
from ...models.gaussian_nb import GaussianNB
from ...models.bayesian_ridge import BayesianRidge
from ...models.res_net import ResNetCV
from ...models.se_res_net import SEResNet
from ...models.alex_net import AlexNet
from ...models.vgg import VGG
from ...models.efficient_net import EfficientNet
from ...models.wrn import WRN
from ...models.reg_net import RegNet
from ...models.sc_net import SCNet
from ...models.pnas_net import PNASNet
from ...models.res_next import ResNeXt

sys.path.append('../../util')
from ...util.constants import logger
from ...util.datasets import convert_dataset_to_numpy

DATASET = None
TEMP_X = None
TEMP_Y = None
MODEL_TYPE = None
CV = 5
RESULTS = None
RANKING = 'f1'
AVERAGE = 'micro'
SCORING = RANKING + '_' + AVERAGE

def finetune(dataset, strategy: str='random', models: list=['decision_tree'], params: Dict=dict(), max_evals: int=32, cv: int=5, model_type: str='C', ranking='f1', average='micro'):
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
    global DATASET
    global TEMP_X
    global TEMP_Y
    global MODEL_TYPE
    global CV 
    global RESULTS
    global RANKING
    global AVERAGE
    global SCORING
    DATASET = dataset
    if not set(models).isdisjoint(SKLEARN_MODELS):
        TEMP_X, TEMP_Y = convert_dataset_to_numpy(dataset, shuffle=True)
        TEMP_X = TEMP_X.reshape(len(dataset),-1)
    MODEL_TYPE = model_type
    CV = cv
    RESULTS = pd.DataFrame(columns=['model', ranking, 'hyperparams', 'time'])
    
    RANKING = ranking
    if SCORING != 'accuracy':
        SCORING = RANKING + '_' + AVERAGE
    
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
        
    DATASET = None
    MODEL_TYPE = None
    CV = 5
    RESULTS = None    
    RANKING = 'f1'
    AVERAGE = 'micro'
    SCORING = RANKING + '_' + AVERAGE
    
    #if not isinstance(best_model, SklearnModel):
    #    logger.info('Training details:')
    #    best_model.plot_metrics()
        
    #return best_model

def _objective(params):
    global CV
    global RESULTS
    global RANKING
    global SCORING
    start = time.time()
    model = params['model']
    del params['model']
    model = _get_model(model, params)
        
    clear_output()
    display(RESULTS)
    
    logger.info("Running cross validation for: " + model.get_name() + "...")
    score = model.cross_val_score(cv=CV, scoring=SCORING).mean() 
            
    elapsed = time.time() - start
    
    clear_output()
    RESULTS.loc[len(RESULTS)] = [model.get_name(), score, model.get_params(), elapsed]
    RESULTS.sort_values(by=RANKING, ascending=False, ignore_index=True, inplace=True)
    display(RESULTS)
    
    # clear cache
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'loss': -score, 'status': STATUS_OK, 'elapsed': elapsed}

def _get_model(model, params=dict()):
    global DATASET    
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
    elif model == 'res_net':
        return ResNetCV(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'se_res_net':
        return SEResNet(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'res_next':
        return ResNeXt(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'alex_net':
        return AlexNet(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'vgg':
        return VGG(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'efficient_net':
        return EfficientNet(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'wrn':
        return WRN(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'reg_net':
        return RegNet(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'sc_net':
        return SCNet(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    elif model == 'pnas_net':
        return PNASNet(DATASET, MODEL_TYPE, params=params, metric_average=AVERAGE)
    else:
        logger.warn(f'Skipping model. Unknown model: {model}')

def _get_search_spaces(models: list, params):
    search_space_list = []
    for model in models:
        if model not in params:
            if model in std_search_space:
                model_space = std_search_space[model].copy()
            else:
                raise ValueError(f'Unknown model: {model}')
        else:
            model_space = params[model]
            model_space['model'] = model
        search_space_list.append(model_space)    

    search_space_model = hp.choice('classifier_type', search_space_list)
    return search_space_model