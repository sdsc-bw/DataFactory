'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset

from .model import PytorchCVModel

class VGG(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict()):
        self.std_in_sizes = [(224, 224)]
        
        ############## process arch params #################
        self.num_layers = params.get('n_layers', 11)
        self.batch_norm = params.get('version', 'a')
        ############## process arch params #################
        
        super(VGG, self).__init__(dataset, model_type, params)
        
        self.name = "VGG"
        self.id = "vgg"
        
    def _init_model(self):
        bn = 'bn_' if self.batch_norm else ''
        self.model = ptcv_get_model(bn + "vgg" + str(self.num_layers), pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)