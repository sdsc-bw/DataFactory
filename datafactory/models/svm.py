'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.svm import SVC, SVR

from .model import SklearnModel

class SVM(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(SVM, self).__init__(X, y, mtype, params)
        if self.mtype == 'C':
            self.model = SVC(**params)
        elif self.mtype == 'R':
            self.model = SVR(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "SVM"
        self.id = "svm"