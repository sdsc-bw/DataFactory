'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import *

from .constants import logger

F1_SCORING = ['f1', 'f1_binary', 'f1_micro', 'f1_weighted', 'f1_macro', 'f1_samples']
PRECISION_SCORING = ['precision', 'precision_binary', 'precision_micro', 'precision_weighted', 'precision_macro', 'precision_samples']
RECALL_SCORING = ['recall', 'recall_binary', 'recall_micro', 'recall_weighted', 'recall_macro', 'recall_samples']
ACCURACY_SCORING = ['accuracy']
MAE_SCORING = ['mae']
MSE_SCORING = ['mse']
R2_SCORING = ['r2']
EXPLAINED_VARIANCE_SCORING = ['explained_variance']


def contvert_to_sklearn_metrics(metric: str):
    while True:
        if metric in F1_SCORING or metric in PRECISION_SCORING or metric in RECALL_SCORING or metric in ACCURACY_SCORING:
            return metric
        elif metric == 'mse':
            return 'neg_mean_squared_error'
        elif metric == 'mae':
            return 'neg_mean_absolute_error'
        else:
            metric = input(f'Not a valid metric: {metric}. Input other metric: ')

def get_metrics_fastai(metric: str, model_type: str, add_top_5_acc=False):
    metrics_list = []
    
    if model_type == 'C':
        metrics_list.append(accuracy)
        if metric == 'binary':
            metrics_list.append(F1Score())
            metrics_list.append(Precision())
            metrics_list.append(Recall())
        elif metric == 'micro':
            metrics_list.append(F1Score(average='micro'))
            metrics_list.append(Precision(average='micro'))
            metrics_list.append(Recall(average='micro'))
        elif metric == 'macro':
            metrics_list.append(F1Score(average='macro'))
            metrics_list.append(Precision(average='macro'))
            metrics_list.append(Recall(average='macro'))
        elif metric == 'samples':
            metrics_list.append(F1Score(average='samples'))
            metrics_list.append(Precision(average='samples'))
            metrics_list.append(Recall(average='samples'))   
        elif metric == 'weighted':
            metrics_list.append(F1Score(average='weighted'))
            metrics_list.append(Precision(average='weighted'))
            metrics_list.append(Recall(average='weighted'))  
            
        if model_type == 'C' and add_top_5_acc:
            metrics_list.append(top_k_accuracy)
    else: 
        metrics_list.append(mse)
        metrics_list.append(mae)
        metrics_list.append(ExplainedVariance())
        metrics_list.append(R2Score())
    
    return metrics_list

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
        
def evaluate_prediction(y, pred, metric):
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy()
        
    results = {}
    
    for m in metric:
        if m == 'explained_variance':
            results['explained_variance'] = explained_variance_score(y, pred)
        elif m == 'mae':
            results['mae'] = mean_absolute_error(y, pred)
        elif m == 'mse':
            results['mse'] = mean_squared_error(y, pred)
        elif m == 'f1':
            results['f1_'] = f1_score(y, pred, average='weighted')
        elif m == 'f1_weighted':
            results['f1_weighted'] = f1_score(y, pred, average='weighted')
        elif m == 'f1_micro':
            results['f1_micro'] = f1_score(y, pred, average='micro')
        elif m == 'f1_macro':
            results['f1_macro'] = f1_score(y, pred, average='macro')
        elif m == 'precision':
            results['precision_'] = precision_score(y, pred, average='weighted')
        elif m == 'precision_weighted':
            results['precision_weighted'] = precision_score(y, pred, average='weighted')
        elif m == 'precisionf1_micro':
            results['precision_micro'] = precision_score(y, pred, average='micro')
        elif m == 'precision_macro':
            results['precision_macro'] = precision_score(y, pred, average='macro')
        elif m == 'recall':
            results['recall_'] = recall_score(y, pred, average='weighted')
        elif m == 'recall_weighted':
            results['recall_weighted'] = recall_score(y, pred, average='weighted')
        elif m == 'recall_micro':
            results['recall_micro'] = recall_score(y, pred, average='micro')
        elif m == 'recall_macro':
            results['recall_macro'] = recall_score(y, pred, average='macro')
        elif m == 'r2':
            results['r2'] = r2_score(y, pred)
        elif m == 'accuracy':
            results['accuracy'] = accuracy_score(y, pred)
        else:
            logger.warn(f'Unknown metric: {metric}. Skipping metric.')
    
    return results
        
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