'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import FCN  as FCNTsai

from .model import TsaiModel

class FCN(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.arch = FCNTsai
        super(FCN, self).__init__(X, y, model_type, params)
        
        self.name = "FCN"
        self.id = "fcn"