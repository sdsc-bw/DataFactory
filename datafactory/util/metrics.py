'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .constants import logger

F1_SCORING = ['f1', 'f1_binary', 'f1_micro', 'f1_weighted', 'f1_macro', 'f1_samples']
PRECISION_SCORING = ['precision', 'precision_binary', 'precision_micro', 'precision_weighted', 'precision_macro', 'precision_samples']
RECALL_SCORING = ['recall', 'recall_binary', 'recall_micro', 'recall_weighted', 'recall_macro', 'recall_samples']
ACCURACY_SCORING = ['accuracy']


def val_score(y_true: List, y_pred: List, scoring: str):
    if scoring in ACCURACY_SCORING:
        return accuracy_score(y_true, y_pred)
    elif scoring in F1_SCORING:
        average = scoring[3:]
        average = 'binary' if average == '' else average
        return f1_score(y_true, y_pred, average=average)
    elif scoring in PRECISION_SCORING:
        average = scoring[10:]
        average = 'binary' if average == '' else average
        return precision_score(y_true, y_pred, average=average)
    elif scoring in RECALL_SCORING:
        average = scoring[7:]
        average = 'binary' if average == '' else average
        return recall_score(y_true, y_pred, average=average)
    else:
        raise ValueError(f'Unknown scoring: {scoring}')     

def relative_absolute_error(pred, y):
    dis = abs((pred-y)).sum()
    dis2 = abs((y.mean() - y)).sum()
    if dis2 == 0 :
        return 1
    return dis/dis2
    
def get_score(y_pred, y_test, model_type='C'):
    if model_type == 'C':
        score = f1_score(y_test, y_pred, average='weighted')
    elif model_type == 'R':
        score = 1 - relative_absolute_error(y_pred, y_test)
    else:
        raise ValueError(f'Unknown type of model: {model}')
        
def evaluate(X: pd.DataFrame, y: pd.Series, cv: int=5, model_type: str='C') -> Tuple[float, float]:
    """Evaluates a dataset with random forests and f1-scores.
        
    Keyword arguments:
    X -- data
    y -- labels
    cv -- number of random forests
    model_type -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
    Output:
    mean of scores
    and variance of scores
    """
    scores = []
    for i in range(cv):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)
        if model_type == 'C':
            model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            score = f1_score(y_test, predict, average='weighted')
        elif model_type == 'R':
            model = RandomForestRegressor(random_state = rf_seed)
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            score = 1 - relative_absolute_error(predict, y_test)
        else:
            raise ValueError(f'Unknown type of model: {model_type}')
        scores.append(score)
    return np.mean(scores), np.std(scores)