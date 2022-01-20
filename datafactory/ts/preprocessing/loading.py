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

sys.path.append('../../util')
from ...util.constants import logger
from ...util.metrics import evaluate

def load_data(fn: str, sep: str=',', shuffle: bool=True) -> Tuple[pd.DataFrame, pd.Series, float]:    
    """Loads and splits dataset into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    fn -- ?
    sep -- seperator of the dataset to seperate cells
    shuffle -- if data should be shuffled
    Output:
    data
    and targets
    and baseline
    """
    df = load_data(fn, logger=logger) # Here is something wrong
    return split(df, shuffle)

def split_data(df: pd.DataFrame, shuffle=True) -> Tuple[np.array, np.array]:
    """Splits the dataframe into data and targets and computes a baseline with random forests.
        
    Keyword arguments:
    df -- dataframe
    shuffle -- if data should be shuffled
    Output:
    data
    and targets
    and baseline
    """
    if shuffle:
        df = sklearn.utils.shuffle(df)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    baseline, _ = evaluate(X, y)
    return X, y, baseline