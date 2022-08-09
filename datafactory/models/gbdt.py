'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .model import SklearnModel

class GBDT(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        super(GBDT, self).__init__(X, y, model_type, params)
        if self.model_type == 'C':
            self.arch = HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(**params)
        elif self.model_type == 'R':
            self.arch = HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "GBDT"
        self.id = "gbdt"