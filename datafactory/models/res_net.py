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

from .model import TsaiModel, PytorchCVModel

## TODO rename to ResNetTS
class ResNet(TsaiModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        self.arch = ResNetTsai
        super(ResNet, self).__init__(X, y, mtype, params)
        
        self.name = "ResNet"
        self.id = "res_net"
        
class ResNetCV(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, mtype: str, params:Dict=dict()):
        
        ############## process arch params #################
        self.num_layers = params.get('num_layers', 10)
        self.pretrained = params.get('pretrained', False)
        self.num_classes = len(dataset.classes)
        dataset_shape = dataset[0][0].shape
        self.in_channels = dataset_shape[0]
        self.in_size =dataset_shape[1], dataset_shape[2]
        ############## process arch params #################
        
        self._init_model()
        super(ResNetCV, self).__init__(dataset, mtype, params)
        
        self.name = "ResNet"
        self.id = "res_net"
        
    def _init_model(self):
        self.model = ptcv_get_model("resnet" + str(self.num_layers), pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)