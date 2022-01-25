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

def load_dataset_from_file(file_path_or_buffer: str, sep: str=',', shuffle: bool=True, transform: List=None) -> Tuple[pd.DataFrame, pd.Series, float]:    
    """Loads and splits dataset into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    file_path_or_buffer -- path to dataset or valid xml string or url
    sep -- seperator of the dataset to seperate cells if csv-file
    shuffle -- if data should be shuffled
    Output:
    data
    and targets
    and baseline
    """
    
    #TODO extend for different file formats
    if file_path_or_buffer.endswith('.csv'):
        df = pd.read_csv(file_path_or_buffer, sep=sep, index_col=[0])
    elif file_path_or_buffer.endswith('.xml'):
        df = pd.read_xml(file_path_or_buffer)
        
    if transform:
        df = apply_transforms(df, transform)
        
    if shuffle:
        df = sklearn.utils.shuffle(df)
        df.reset_index(drop=True, inplace=True)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y