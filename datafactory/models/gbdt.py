import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .model import SklearnModel

class GBDT(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(GBDT, self).__init__(X, y, mtype, params)
        if self.mtype == 'C':
            self.model = HistGradientBoostingClassifier(**params)
        elif self.mtype == 'R':
            self.model = HistGradientBoostingRegressor(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "GBDT"
        self.id = "gbdt"