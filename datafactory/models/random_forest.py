'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .model import SklearnModel

class RandomForest(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(RandomForest, self).__init__(X, y, model_type, params)
        if self.model_type == 'C':
            self.model = RandomForestClassifier(**params)
        elif self.model_type == 'R':
            self.model = RandomForestRegressor(**params)
        else:
            logger.error('Unknown type of model')
            
        self.name= "Random Forest"
        self.id = "random_forest"