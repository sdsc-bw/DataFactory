'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tsai.all import ResNet  as ResNetTsai
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset
import sys
import random

from .model import TsaiModel, PytorchCVModel

sys.path.append('../util')
from ..util.constants import logger

## TODO rename to ResNetTS
class ResNet(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, model_type: str, params:Dict=dict()):
        self.arch = ResNetTsai
        super(ResNet, self).__init__(X, y, model_type, params)
        
        self.name = "ResNet"
        self.id = "res_net"
        
class ResNetCV(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict()):
        
        ############## process arch params #################
        self.num_layers = params.get('n_layers', 10)
        self.down_sampling = params.get('down_sampling', False)
        ############## process arch params #################
        
        super(ResNetCV, self).__init__(dataset, model_type, params)
        
        self.name = "ResNet"
        self.id = "res_net"
        
    def _init_model(self):
        if self.down_sampling:
            if self.num_layers == 10 or self.num_layers == 18:
                self.model = ptcv_get_model("resneta" + str(self.num_layers), pretrained=self.pretrained, 
                                            num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
            elif self.num_layers == 14:
                self.model = ptcv_get_model("resnetabc" + str(self.num_layers) + "b", pretrained=self.pretrained, 
                                            num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
            elif self.num_layers == 50 or self.num_layers == 101 or self.num_layers == 152:
                self.model = ptcv_get_model("resneta" + str(self.num_layers) + "b", pretrained=self.pretrained, 
                                            num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
            else:
                self.down_sampling = False
                self.params['down_sampling'] = False
        else:
            self.model = ptcv_get_model("resnet" + str(self.num_layers), pretrained=self.pretrained, 
                                        num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
        self.model.to(self.device)