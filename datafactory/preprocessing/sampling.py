import pandas as pd
import numpy as np
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import * 
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import sys

sys.path.append('../util')
from ..util.constants import logger

def sampling_up(X: pd.DataFrame, y: pd.Series, strategy: str = 'SMOTE', random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Samples down dataset based on given sampling strategy.
    Keyword arguments:
    X -- data
    y -- targets
    strategy -- sampling strategy, should be in ['smote', 'random', 'borderline', 'adasyn', 'kmeanssmote']
    random_state -- controlls the randomization of the algorithm
        
    Output:
    up sampled data
    and up sampled targets
    """
    logger.info(f'Start to apply upsampling strategy: {strategy}...')
    if strategy == 'smote':
        usa = SMOTE(sampling_strategy='auto', random_state=random_state)
    elif strategy == 'random':
        usa = RandomOverSampler(sampling_strategy='auto', random_state=random_state)
    elif strategy == 'borderline':
        usa = BorderlineSMOTE(sampling_strategy='auto', random_state=random_state)
    elif strategy == 'adasyn':
        usa = ADASYN(sampling_strategy='auto', random_state=random_state)
    elif strategy == 'kmeanssmote':
        usa = KMeansSMOTE(sampling_strategy='auto', random_state=random_state)
    else:
        logger.warn('Unrecognized upsampling strategy. Use SMOTE instead')
        usa = SMOTE(sampling_strategy='auto', random_state=random_state)
    res_x, res_y = usa.fit_resample(X, y)
    if type(res_x) == np.ndarray:
        res_x = pd.DataFrame(res_x, columns = X.columns)
        res_y = pd.Series(res_y, name = y.name)
    logger.info('...End with upsampling')
    return res_x, res_y

def sampling_down(X: pd.DataFrame, y: pd.Series, strategy: str = 'SMOTE', random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Samples down dataset based on given sampling strategy.
    Keyword arguments:
    X -- data
    y -- targets
    strategy -- sampling strategy, should be in ['cluster', 'random', 'nearmiss1', 'nearmiss2', 'nearmiss3', 'tomek', 'enn', 'repeatenn', 'allknn', 'condensednn']
    random_state -- controlls the randomization of the algorithm
        
    Output:
    down sampled data
    and down sampled targets
    """
    logger.info(f'Start to apply downsampling strategy: {strategy}...')
    if strategy == 'cluster':
        dsa = ClusterCentroids(sampling_strategy = 'auto', random_state = random_state)
    elif strategy == 'random':
        dsa = RandomUnderSampler(sampling_strategy = 'auto', random_state = random_state)
    elif strategy == 'nearmiss1':
        dsa = NearMiss(sampling_strategy = 'auto', version = 1)
    elif strategy == 'nearmiss2':
        dsa = NearMiss(sampling_strategy = 'auto', version = 2)
    elif strategy == 'nearmiss3':
        dsa = NearMiss(sampling_strategy = 'auto', version = 3)
    elif strategy == 'tomek':
        dsa = TomekLinks(sampling_strategy = 'auto')
    elif strategy == 'enn':
        dsa = EditedNearestNeighbours(sampling_strategy = 'auto')
    elif strategy == 'repeatenn':
        dsa = RepeatedEditedNearestNeighbours(sampling_strategy = 'auto')
    elif strategy == 'allknn':
        dsa = AllKNN(sampling_strategy = 'auto')
    elif strategy == 'condensednn':
        dsa = CondensedNearestNeighbour(sampling_strategy = 'auto', random_state = random_state)
    else:
        logger.warn('Unrecognized downsampling strategy. Use TOMEK instead')
        dsa = TomekLinks(sampling_strategy = 'auto')
    res_x, res_y = dsa.fit_resample(X, y)
    if type(res_x) == np.ndarray:
        res_x = pd.DataFrame(res_x, columns = X.columns)
        res_y = pd.Series(res_y, name = y.name)
    logger.info('...End with downsampling')
    return res_x, res_y

def sampling_combine(X: pd.DataFrame, y: pd.Series, strategy: str = 'SMOTE', random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Samples dataset based on given sampling strategy.
    Keyword arguments:
    X -- data
    y -- tagets
    strategy -- sampling strategy, should be in ['smoteenn', 'smotetomek']
        
    Output:
    combine sampled data
    and combine sampled targets
    """
    logger.info(f'Start to apply combine sampling strategy: {strategy}...')
    if strategy == 'smoteenn':
        csa = SMOTEENN(sampling_strategy = 'auto', random_state = random_state)
    elif strategy == 'smotetomek':
        csa = SMOTETomek(sampling_strategy = 'auto', random_state = random_state)
    else:
        logger.warn('Unrecognized downsampling strategy... Use SMOTEENN instead')
        csa = SMOTEENN(sampling_strategy = 'auto', random_state = random_state)
    res_x, res_y = csa.fit_resample(X, y)
    if type(res_x) == np.ndarray:
        res_x = pd.DataFrame(res_x, columns = X.columns)
        res_y = pd.Series(res_y, name = y.name)
    logger.info('...End with combine sampling')
    return res_x, res_y