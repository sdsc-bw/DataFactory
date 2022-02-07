'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from .model import SklearnModel

class AdaBoost(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(AdaBoost, self).__init__(X, y, model_type, params)
        if self.model_type == 'C':
            self.model = AdaBoostClassifier(**params)
        elif self.model_type == 'R':
            self.model = AdaBoostRegressor(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "AdaBoost"
        self.id = "ada_boost"