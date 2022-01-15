import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import LSTM  as LSTMTsai

from .model import TsaiModel

class LSTM(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = LSTMTsai
        super(LSTM, self).__init__(X, y, mtype, params)
        
        self.name = "LSTM"
        self.id = "lstm"