'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from tsai.all import *
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .constants import logger

def get_optimizer_fastai(optimizer:str):
    if optimizer == 'adam':
        return Adam
    elif optimizer == 'r_adam':
        return RAdam
    elif optimizer == 'qh_adam':
        return QHAdam
    elif optimizer == 'sgd':
        return SGD
    elif optimizer == 'rms_prop':
        return RMSProp    
    elif optimizer == 'larc':
        return Larc
    elif optimizer == 'lamb':
        return Lamb
    else:
        logger.warn(f'Unknown optimizer: {optimizer}. Using Adam instead')
        return Adam
