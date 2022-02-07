'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import TCN as TCNTsai

from .model import TsaiModel

class TCN(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        self.arch = TCNTsai
        super(TCN, self).__init__(X, y, model_type, params)
        
        self.name = "TCN"
        self.id = "tcn"