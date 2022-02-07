'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset

from .model import PytorchCVModel

class SEResNet(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict()):
        
        ############## process arch params #################
        self.num_layers = params.get('n_layers', 10)
        ############## process arch params #################
        
        super(SEResNet, self).__init__(dataset, model_type, params)
        
        self.name = "SEResNet"
        self.id = "se_res_net"
        
    def _init_model(self):
        self.model = ptcv_get_model("seresnet" + str(self.num_layers), pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
        self.model.to(self.device)