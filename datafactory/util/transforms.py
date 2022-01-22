'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .constants import logger

def get_transforms_ts(transform: List, params: Dict=dict()):    
    ## TODO implement
    pass

def apply_transforms_ts(df: pd.DataFrame, params: Dict=dict()):
    ## TODO implement
    pass

def get_transforms_cv(transform: List, params: Dict=dict()):
    transform_compose = []
    if transform:
        for i in transform:
            p = params.get(i, None)
            transform_compose.append(_get_transform(i, p))
    else:
        transform_compose.append(_get_transform('to_tensor', None))
    return transforms.Compose(transform_compose)
        
        
def _get_transform_cv(transform, params=None):
    if transform == 'center_crop':
        return transforms.CenterCrop(**params) if params else transforms.CenterCrop(10)
    elif transform == 'five_crop':
        return transforms.FiveCrop(**params) if params else transforms.FiveCrop(10)
    elif transform == 'random_crop':
        return transforms.RandomCrop(**params) if params else transforms.RandomCrop(10)
    elif transform == 'pad':
        return transforms.Pad(**params) if params else transforms.Pad(5)
    elif transform == 'resize':
        return transforms.Resize(**params) if params else transforms.Resize(144)
    elif transform == 'random_rotation':
        return transforms.RandomRotation(**params) if params else transforms.RandomRotation(20)
    elif transform == 'color_jitter':
        return transforms.ColorJitter(**params)  if params else transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    elif transform == 'grayscale':
        return transforms.Grayscale(**params) if params else transforms.Grayscale()
    elif transform == 'normalize':
        transforms.Normalize(**params) if params else transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    elif transform == 'to_tensor':
        return transforms.PILToTensor(**params) if params else transforms.ToTensor()
    else:
        logger.error(f'Unknown transformation: {transform}')