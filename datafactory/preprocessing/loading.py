'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sklearn.utils
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import sys

sys.path.append('../util')
from ..util.constants import logger
from ..util.metrics import relative_absolute_error

def load_data(fn: str, sep: str=',', shuffle: bool=True) -> Tuple[pd.DataFrame, pd.Series, float]:    
    """Loads and splits dataset into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    fn -- ?
    sep -- seperator of the dataset to seperate cells
    shuffle -- if data should be shuffled
    Output:
    data
    and targets
    and baseline
    """
    df = load_data(fn, logger=logger) # Here is something wrong
    return split(df, shuffle)

def split_data(df: pd.DataFrame, shuffle=True) -> Tuple[np.array, np.array]:
    """Splits the dataframe into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    df -- dataframe
    shuffle -- if data should be shuffled
    Output:
    data
    and targets
    and baseline
    """
    if shuffle:
        df = sklearn.utils.shuffle(df)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    baseline, _ = evaluate(X, y)
    return X, y, baseline

def evaluate(X: pd.DataFrame, y: pd.Series, cv: int=5, mtype: str='C') -> Tuple[float, float]:
    """Evaluates a dataset with random forests and f1-scores.
        
    Keyword arguments:
    X -- data
    y -- labels
    cv -- number of random forests
    mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
    Output:
    mean of scores
    and variance of scores
    """
    scores = []
    for i in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        if mtype == 'C':
            model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            score = f1_score(y_test, predict, average='weighted')
        elif mtype == 'R':
            model = RandomForestRegressor(random_state = rf_seed)
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            score = 1 - relative_absolute_error(predict, y_test)
        else:
            logger.error('Unknown type of task')
        scores.append(score)
    return np.mean(scores), np.std(scores)