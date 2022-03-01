'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset
import sys

from .model import PytorchCVModel

sys.path.append('../util')
from ..util.constants import logger

class EfficientNet(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.std_in_sizes = [(224, 224), (240, 240), (260, 260), (300, 300), (380, 380), 
                                   (456, 456), (528, 528), (600, 600), (672, 672)]
        
        super(EfficientNet, self).__init__(dataset, model_type, params)
        
        ############## process arch params #################
        if self.in_size == (224, 224):
            self.version = 'b0'
        elif self.in_size == (240, 240):
            self.version = 'b1'
        elif self.in_size == (260, 260):
            self.version = 'b2'
        elif self.in_size == (300, 300):
            self.version = 'b3'
        elif self.in_size == (380, 380):
            self.version = 'b4'
        elif self.in_size == (456, 456):
            self.version = 'b5'
        elif self.in_size == (528, 528):
            self.version = 'b6'
        elif self.in_size == (600, 600):
            self.version = 'b7'
        elif self.in_size == (672, 672):
            self.version = 'b8'
        else:
            logger.error(f'Unsupported input size: {self.in_size}')            
        ############## process arch params #################
        
        self.name = "EfficientNet"
        self.id = "efficient_net"
        
    def _init_model(self):
        self.model = ptcv_get_model("efficientnet_" + self.version, pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
       