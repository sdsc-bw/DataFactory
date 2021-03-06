'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .model import PytorchCVModel

class AlexNet(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params: Dict=dict(), metric_average: str='micro'):
        self.std_in_sizes = [(224, 224)]
        
        super(AlexNet, self).__init__(dataset, model_type, params)
        
        ############## process arch params #################
        self.version = params.get('version', 'a')
        ############## process arch params #################
        
        self.name = "AlexNet"
        self.id = "alex_net"
        
    def _init_model(self):
        self.model = ptcv_get_model("alexnet", pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels, version=self.version)
        