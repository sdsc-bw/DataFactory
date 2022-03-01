'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset
import random

from .model import PytorchCVModel

class ResNeXt(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.std_in_sizes = [(224, 224)]
        
        ############## process arch params #################
        self.num_blocks = params.get('n_blocks', 14)
        self.cardinality = params.get('cardinality', 16)
        self.bottleneck_width = params.get('bottleneck_width', 2)
        if self.num_blocks == 14 or self.num_blocks == 26:
            if self.cardinality == 16 and self.bottleneck_width == 2:
                self.bottleneck_width = 4
            if self.cardinality == 64:
                self.cardinality = random.choice([16, 32])
        elif self.num_blocks == 38 or self.num_blocks == 50:
            if self.cardinality == 16 or self.cardinality == 64:
                self.cardinality = 32
            if self.cardinality == 32 and self.bottleneck_width == 2:
                self.bottleneck_width = 4
        elif self.num_blocks == 101:
            if self.cardinality == 16:
                self.cardinality = random.choice([32, 64])
            self.bottleneck_width = 4
            
        ############## process arch params #################
        
        super(ResNeXt, self).__init__(dataset, model_type, params)
        
        self.name = "ResNeXt"
        self.id = "res_next"
        
    def _init_model(self):
        self.model = ptcv_get_model(f"resnext{str(self.num_blocks)}_{self.cardinality}x{self.bottleneck_width}d", pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)