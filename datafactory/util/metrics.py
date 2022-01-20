'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .constants import logger

def relative_absolute_error(pred, y):
    dis = abs((pred-y)).sum()
    dis2 = abs((y.mean() - y)).sum()
    if dis2 == 0 :
        return 1
    return dis/dis2
    
def get_score(y_pred, y_test, mtype='C'):
    if mtype == 'C':
        score = f1_score(y_test, y_pred, average='weighted')
    elif mtype == 'R':
        score = 1 - relative_absolute_error(y_pred, y_test)
    else:
        logger.error('Unknown type of model')
        
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