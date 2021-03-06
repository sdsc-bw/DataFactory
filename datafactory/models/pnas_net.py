'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset

from .model import PytorchCVModel

class PNASNet(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict(), metric_average: str='micro'):      
        self.available_in_sizes = [(224, 224), (331, 331)]
        
        super(PNASNet, self).__init__(dataset, model_type, params)
        
        self.name = "PNASNet"
        self.id = "pnas_net"
        
    def _init_model(self):
        self.model = ptcv_get_model("pnasnet5large", pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)