'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset
import sys

from .model import TsaiModel, PytorchCVModel

sys.path.append('../util')
from ..util.constants import logger
        
class SCNet(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict()):
        self.available_in_sizes = [(224, 224), (256, 256)]
        ############## process arch params #################
        self.num_layers = params.get('n_layers', 50)
        self.down_sampling = params.get('down_sampling', False)
        ############## process arch params #################
        
        super(SCNet, self).__init__(dataset, model_type, params)
        
        self.name = "SCNet"
        self.id = "sc_net"
        
    def _init_model(self):
        a = 'a' if self.down_sampling else ''
            
        self.model = ptcv_get_model("scnet" + a + str(self.num_layers), pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
        self.model.to(self.device)