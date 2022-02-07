'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.naive_bayes import GaussianNB as GaussianNBSklearn

from .model import SklearnModel

class GaussianNB(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        super(GaussianNB, self).__init__(X, y, model_type, params)
        if self.model_type == 'C':
            self.model = GaussianNBSklearn(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "Gaussian NB"
        self.id = "gaussian_nb"