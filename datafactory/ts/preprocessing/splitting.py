'''
Copyright (c) Smart Data Solution Center Baden-WÃ¼rttemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import sys

def split_X_y(data, target, strategy='last', n=1):
    if strategy == 'last':
        X = data.copy(deep=True)
       
        for i in range(n-1, 0, -1):
            X['Last_' + str(i) + '_' + target] = X[target].shift(i)
        
        y = X[target].shift(-1)
        
        # move current value to the end to the previous values 
        X.insert(len(X.columns)-1, 'Current_' + target, X.pop(target))
        
        y = y[n-1:]
        X = X[n-1:]
       
        y = y[:-n]
        X = X[:-n]
    elif strategy == 'rolling_window':
        X = data.copy(deep=True)
        tmp = data[[target]].shift(-1)
        # TODO edit
        #X['Last_n_mean_' + target] = tmp[target] - tmp.rolling(n).mean()[target]
        X['Last_n_mean_' + target] = tmp.rolling(n).mean()[target]
        y = data[target]
                
        y = y[:-n]
        X = X[:-n]
        
    else:
        logger.info(f'Unrecognized strategy. Using last instead.')
        
        X = data.copy(deep=True)
       
        for i in range(n-1, 0, -1):
            X['Last_' + str(i) + '_' + target] = X[target].shift(i)
        
        y = X[target].shift(-1)
        X.insert(len(X.columns)-1, target, X.pop(target))
       
        y = y[:-n]
        X = X[:-n]
    
    return X, y