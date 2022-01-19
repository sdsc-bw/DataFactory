'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .model import SklearnModel

class KNN(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(KNN, self).__init__(X, y, mtype, params)
        if self.mtype == 'C':
            self.model = KNeighborsClassifier(**params)
        elif self.mtype == 'R':
            self.model = KNeighborsRegressor(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "KNN"
        self.id = "knn"