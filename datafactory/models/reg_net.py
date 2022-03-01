'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.utils.data import Dataset

from .model import PytorchCVModel

class RegNet(PytorchCVModel):
    
    def __init__(self, dataset: Dataset, model_type: str, params:Dict=dict(), metric_average: str='micro'):
        self.std_in_sizes = [(224, 224)]
        
        ############## process arch params #################
        self.version = params.get('version', 'x') # SE = squeeze-and-excitation 
        self.num_mf = params.get('n_mf', 200) # MF = million flop
        ############## process arch params #################
        
        super(RegNet, self).__init__(dataset, model_type, params)
        
        self.name = "RegNet"
        self.id = "reg_net"
        
    def _init_model(self):
        
        if self.num_mf == 200:
            mf = '002'
        elif self.num_mf == 400:
            mf = '004'
        elif self.num_mf == 600:
            mf = '006'
        elif self.num_mf == 800:
            mf = '008'
        elif self.num_mf == 1600:
            mf = '016'
        elif self.num_mf == 3200:
            mf = '032'
        elif self.num_mf == 4000:
            mf = '040'
        elif self.num_mf == 6400:
            mf = '064'
        elif self.num_mf == 8000:
            mf = '080'
        elif self.num_mf == 12000:
            mf = '120'    
        elif self.num_mf == 16000:
            mf = '160'
        elif self.num_mf == 32000:
            mf = '320'
        self.model = ptcv_get_model("regnet" + self.version + mf, pretrained=self.pretrained, 
                                    num_classes=self.num_classes, in_size=self.in_size, in_channels=self.in_channels)
        self.model.to(self.device)