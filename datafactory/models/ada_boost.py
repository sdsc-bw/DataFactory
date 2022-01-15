import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from .model import SklearnModel

class AdaBoost(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(AdaBoost, self).__init__(X, y, mtype, params)
        if self.mtype == 'C':
            self.model = AdaBoostClassifier(**params)
        elif self.mtype == 'R':
            self.model = AdaBoostRegressor(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "AdaBoost"
        self.id = "ada_boost"