'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import sys

def add_time_as_columns(data):
    data['Month'] = data.index.month
    data['Day of the Week'] = data.index.dayofweek
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour
    data['Minute'] = data.index.minute
    data['Second'] = data.index.second
    return data