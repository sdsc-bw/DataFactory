'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import GRU  as GRUTsai

from .model import TsaiModel

class GRU(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.arch = GRUTsai
        super(GRU, self).__init__(X, y, model_type, params)
        
        self.name = "GRU"
        self.id = "gru"