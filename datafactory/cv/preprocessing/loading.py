'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import numpy as np
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sklearn
import sys

sys.path.append('../../util')
from ...util.constants import logger
from ...util.metrics import evaluate
from ...util.datasets import convert_dataset_to_numpy, CSVDataset, CSVFolderDataset
from ...util.transforms import get_transforms_cv

def load_dataset_from_file(file_path: str, root_dir: str=None, shuffle: bool=True, transform: List=['to_tensor'], 
                           transform_params: Dict=dict(), batch_size: int=64) -> Tuple[np.array, np.array, float]:
    """Loads and splits dataset into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    file_path -- file path to the dataset or to file with file_paths
    root_dir -- root directory of the files in 'file_path'
    shuffle -- if dataset should be shuffled
    transforms -- transformations that should be applied on the data, should always contain 'to_tensor'
    transform_params -- params for transformations listed in 'transforms'
    Output:
    data
    and targets
    and baseline
    """
    transform = get_transforms_cv(transform, transform_params)
        
    #TODO extend for different file formats    
    if file_path.endswith('.csv') and root_dir:
        return CSVFolderDataset(file_path, root_dir=root_dir, transform=transform)
    elif file_path.endswith('.csv'):
        return CSVDataset(file_path, transform=transform)
    else:
        return datasets.ImageFolder(file_path, transform=transform)