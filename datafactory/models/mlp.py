import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import MLP  as MLPTsai

from .model import TsaiModel

class MLP(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = MLPTsai
        super(MLP, self).__init__(X, y, mtype, params)
        
        self.name = "MLP"
        self.id = "mlp"