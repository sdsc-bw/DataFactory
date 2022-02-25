'''
Copyright (c) Smart Data Solution Center Baden-WÃ¼rttemberg 2021,
All rights reserved.
'''

import pandas as pd
from scipy.stats import iqr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import sys

sys.path.append('../../util')
from ...util.constants import logger, bcolors

def outlier_detection_dataframe(df: pd.DataFrame) -> pd.Series:
    """Outlier detection of a given dataframe.
        
    Keyword arguments:
    
    df -- dataframe

    Output:
    out: output pd.Series that signify if each item is outlier or not
    """
    
    logger.info(f'Start to detect outlier for the whole data set...')
    print('#'*30)
    print('Outliear detection')
    print('#'*30)
    
    if df.shape[1]>= 40:
        print('Detect outlier with strategy: high demension')
        out = outlier_detection_high_dimension(df)
    else:
        print('Detect outlier with strategy: density')
        out = outlier_detection_density(df)

    logger.info(f'...End with outlier detection, {bcolors.HEADER}{bcolors.BOLD}{out.sum()}{bcolors.ENDC} outliers found')
    print(f'{bcolors.HEADER}{bcolors.BOLD}{out.sum()}{bcolors.ENDC} outliers found')

    return out

def outlier_detection_feature(col: pd.Series) -> pd.Series:
    """Outlier detection of a given column.
        
    Keyword arguments:
    col -- column
    Output:
    outlier of given feature
    """
    logger.info(f'Start to detect outlier for given feature {col.name} with 3 IQR strategy...')
    v_iqr = iqr(col)
    v_mean = col.mean()
    ceiling = v_mean + 3*v_iqr
    floor = v_mean - 3*v_iqr
    out = col.map(lambda x: x>ceiling or x<floor)
    logger.info(f'...End with outlier detection, {out.sum()} outliers found')
    return out

def outlier_detection_high_dimension(df: pd.DataFrame) -> pd.Series:
    """High dimension outlier detection of a given dataframe with a random forest.
        
    Keyword arguments:
    df -- dataframe
    Output:
    outlier of given dataframe
        """
    clf = IsolationForest(n_estimators=20, warm_start=True)
    out = pd.Series(clf.fit_predict(df), index = df.index)
    out = out.map(lambda x: x == -1)
    return out

def outlier_detection_density(df: pd.DataFrame) -> pd.Series:
    """Density-based outlier detection of a given dataframe.
        
    Keyword arguments:
    df -- dataframe
    Output:
    outlier of given dataframe
    """
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    out = pd.Series(clf.fit_predict(df), index = df.index)
    out = out.map(lambda x: x == -1)
    return out
