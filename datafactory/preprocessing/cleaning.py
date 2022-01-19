'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import sys

sys.path.append('../util')
from ..util.constants import logger

def clean_data(data: pd.DataFrame, strategy: str='model') -> pd.DataFrame:
    """Clean dataset of INF- and NAN-values.
    
    Keyword arguments:
    dat -- dataframe
    strategy -- cleaning strategy, should be in ['model', 'mean', 'median', 'most_frequent', 'constant']
    Output:
    data -- cleaned dataframe
    """
    if data.empty:
        return data
    logger.info('Start to clean the given dataframe...')
    logger.info('Number of INF- and NAN-values are: (%d, %d)' % ((data == np.inf).sum().sum(), data.isna().sum().sum()))
    logger.info('Set type to float32 at first && deal with INF')
    data = data.astype(np.float32)
    data = data.replace([np.inf, -np.inf], np.nan)
    logger.info('Remove columns with half of NAN-values')
    data = data.dropna(axis=1, thresh=data.shape[0] * .5)
    logger.info('Remove constant columns')
    data = data.loc[:, (data != data.iloc[0]).any()]

    if data.isna().sum().sum() > 0:
        logger.info('Start to fill the columns with NAN-values...')
        # imp = IterativeImputer(max_iter=10, random_state=0)
        if strategy == 'model':
            imp = IterativeImputer(max_iter=10, random_state=0)
        elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
            imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        else:
            logger.warn(f'Unrecognized strategy {strategy}. Use mean instead')
        # data = data.fillna(data.mean())
        tmp = imp.fit_transform(data)
        if tmp.shape[1] != data.shape[1]:
            logger.warn(f'Error appeared while fitting. Use constant filling instead')
            tmp = data.fillna(0)
        data = pd.DataFrame(tmp, columns=data.columns, index=data.index)
    #logger.info('Remove rows with any nan in the end')
    #data = data.dropna(axis=0, how='any')
    logger.info('...End with Data cleaning, number of INF- and NAN-values are now: (%d, %d)' 
                     % ((data == np.inf).sum().sum(), data.isna().sum().sum()))
    #data = data.reset_index(drop=True)
    return data