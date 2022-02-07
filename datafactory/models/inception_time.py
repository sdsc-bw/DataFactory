'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import InceptionTime  as InceptionTimeTsai

from .model import TsaiModel

class InceptionTime(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        self.arch = InceptionTimeTsai
        super(InceptionTime, self).__init__(X, y, model_type, params)
        
        self.name = "InceptionTime"
        self.id = "inception_time"