'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.linear_model import BayesianRidge as BayesianRidgeSklearn

from .model import SklearnModel

class BayesianRidge(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(BayesianRidge, self).__init__(X, y, model_type, params)
        if self.model_type == 'R':
            self.model = BayesianRidgeSklearn(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "Bayesian Ridge"
        self.id = "bayesian_ridge"