'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import sys
import numpy as np
import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import TimeSeriesSplit

from .model import Model, SklearnModel

sys.path.append('../util')
from ..util.constants import logger
from ..util.metrics import evaluate_prediction

class Baseline(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(Baseline, self).__init__(X, y, model_type, params)
        if self.model_type == 'C':
            self.arch = DummyClassifier
            self.model = DummyClassifier(**params)
        elif self.model_type == 'R':
            self.arch = DummyRegressor
            self.model = DummyRegressor(**params)
        else:
            logger.error('Unknown type of model')

        self.name= "Baseline"
        self.id = "baseline"
        
class BaselineTS(Model):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(BaselineTS, self).__init__(model_type)

        # TODO this should be in superclass
        self.X = X
        self.y = y
        self.params = params
        
        self.lags = params.get('lags', 1)
        
        self.name= "Baseline"
        self.id = "baseline_ts"

    def fit(self, X, y):
        pass
    
    def cross_val_score(self, cv=5, scoring='f1_micro'):
        if type(scoring) != list:
            scoring =  [scoring]
        
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = {}
        for train_index, test_index in tscv.split(self.X):
            self._reset_model()
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            self.fit(X_train, y_train)
            pred = self.predict(X_test, test_index)
            
            tmp_scores = evaluate_prediction(y_test, pred, scoring)
            for s in tmp_scores:
                if not s in scores.keys():
                    scores[s] = []
                scores[s].append(tmp_scores[s])
        self._reset_model() 
        
        return scores  
    
    def predict(self, X, y_idx):
        # TODO add prediction for classification
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        pred = self.y.shift(self.lags).iloc[y_idx]
        
        pred = pred.fillna(method='ffill')
        pred = pred.fillna(method='bfill')
        
        pred = pred.to_numpy()
        pred = pred.reshape((-1,))

        return pred
    
    
    def predict_probas(self, X):
        # TODO add prediction for classification
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # TODO where to get y from when X != self.X?
        
        pred = self.y.shift(self.lags)
        
        pred = pred.fillna(method='ffill')
        pred = pred.fillna(method='bfill')
        
        pred = pred.to_numpy()
        pred = pred.reshape((-1,))

        return pred
    