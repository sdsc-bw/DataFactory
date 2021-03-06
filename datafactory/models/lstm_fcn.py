'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import LSTM_FCN

from .model import TsaiModel

class LSTMFCN(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.arch = LSTM_FCN
        super(LSTMFCN, self).__init__(X, y, model_type, params)
        
        self.name = "LSTM-FCN"
        self.id = "lstm_fcn"