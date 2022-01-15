import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import GRU_FCN

from .model import TsaiModel

class GRUFCN(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = GRU_FCN
        super(GRUFCN, self).__init__(X, y, mtype, params)
        
        self.name = "GRU-FCN"
        self.id = "gru_fcn"