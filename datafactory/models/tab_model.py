'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import TabModel as TabModelTsai

from .model import TsaiModel

class TabModel(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.arch = TabModelTsai
        super(TabModel, self).__init__(X, y, model_type, params)
        
        self.name = "TabModel"
        self.id = "tab_model"