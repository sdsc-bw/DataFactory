import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import XceptionTime as XceptionTimeTsai

from .model import TsaiModel

class XceptionTime(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = XceptionTimeTsai
        super(XceptionTime, self).__init__(X, y, mtype, params)
        
        self.name = "XceptionTime"
        self.id = "xception_time"