'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import f1_score
import sklearn.utils
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import sys

sys.path.append('../feature_engineering')
from ..feature_engineering.transforming import apply_transforms

sys.path.append('../../util')
from ...util.constants import logger
from ...util.metrics import evaluate

def load_dataset_from_file(file_path: str, sep: str=',', shuffle: bool=True, transform: List=None) -> Tuple[pd.DataFrame, pd.Series, float]:    
    """Loads and splits dataset into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    file_path -- path to dataset
    sep -- seperator of the dataset to seperate cells
    shuffle -- if data should be shuffled
    Output:
    data
    and targets
    and baseline
    """
    
    #TODO extend for different file formats
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, sep=sep)
        
    if transform:
        df = apply_transforms(df, transform)
        
    if shuffle:
        df = sklearn.utils.shuffle(df)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y