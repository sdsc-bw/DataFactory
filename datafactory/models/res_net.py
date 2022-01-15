import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import ResNet  as ResNetTsai

from .model import TsaiModel

class ResNet(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = ResNetTsai
        super(ResNet, self).__init__(X, y, mtype, params)
        
        self.name = "ResNet"
        self.id = "res_net"