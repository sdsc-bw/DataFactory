import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import LSTM_FCN

from .model import TsaiModel

class LSTMFCN(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = LSTM_FCN
        super(LSTMFCN, self).__init__(X, y, mtype, params)
        
        self.name = "LSTM-FCN"
        self.id = "lstm_fcn"