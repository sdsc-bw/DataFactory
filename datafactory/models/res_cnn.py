'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import ResCNN as ResCNNTsai

from .model import TsaiModel

class ResCNN(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.arch = ResCNNTsai
        super(ResCNN, self).__init__(X, y, model_type, params)
        
        self.name = "ResCNN"
        self.id = "res_cnn"