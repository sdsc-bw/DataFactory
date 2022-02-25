'''
Copyright (c) Smart Data Solution Center Baden-WÃ¼rttemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import sys

sys.path.append('../../util')
from ...util.constants import logger, bcolors

def clean_data(data: pd.DataFrame, strategy: str='model', file = None) -> pd.DataFrame:
    """Clean dataset of INF- and NAN-values.
    
    Keyword arguments:
    dat -- dataframe
    strategy -- cleaning strategy, should be in ['model', 'mean', 'median', 'most_frequent', 'constant']
    file -- file to save the cleaning information
    Output:
    data -- cleaned dataframe
    """
    if data.empty:
        return data
        
    logger.info('Start to clean the given dataframe...')
    
    print('#'*30, file = file)
    print('clean na and inf in the data', file = file)
    print('#'*30, file = file)
    
    print(f'Number of INF- and NAN-values are: ({bcolors.HEADER}{bcolors.BOLD}{(data == np.inf).sum().sum()}{bcolors.ENDC}, {bcolors.HEADER}{bcolors.BOLD}{data.isna().sum().sum()}{bcolors.ENDC})', file = file)
    print('Set type to float32 at first && deal with INF', file = file)
    
    data = data.astype(np.float32)
    data = data.replace([np.inf, -np.inf], np.nan)
    
    print(f'Remove columns with half of NAN-values: {data.columns[(data.isna().sum()/data.shape[0])>=0.5].to_list()}', file = file)
    data = data.dropna(axis=1, thresh=data.shape[0] * .5)
    
    print(f'Remove constant columns: {data.columns[(data == data.iloc[0]).all()].to_list()}', file = file)
    data = data.loc[:, (data != data.iloc[0]).any()]

    if data.isna().sum().sum() > 0:
        
        if strategy == 'model':
            
            imp = IterativeImputer(max_iter=10, random_state=0)
            
        elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
            
            imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
            
        else:
            
            logger.warn(f'Unrecognized strategy {strategy}. Use mean instead')
            
        tmp = imp.fit_transform(data)
        
        if tmp.shape[1] != data.shape[1]:
            logger.warn(f'Error appeared while fitting. Use constant filling instead')
            tmp = data.fillna(0)
            
        data = pd.DataFrame(tmp, columns=data.columns, index=data.index)
        
    #logger.info('Remove rows with any nan in the end')
    #data = data.dropna(axis=0, how='any')
    logger.info(f'...End with Data cleaning, number of INF- and NAN-values are now: ({(data == np.inf).sum().sum()}, {data.isna().sum().sum()})')
    print(f'End with Data cleaning, number of INF- and NAN-values are now: ({bcolors.HEADER}{bcolors.BOLD}{(data == np.inf).sum().sum()}{bcolors.ENDC}, {bcolors.HEADER}{bcolors.BOLD}{data.isna().sum().sum()}{bcolors.ENDC})', file = file)
    #data = data.reset_index(drop=True)
    return data
