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
import copy

sys.path.append('../../util')
from ...util.constants import logger, bcolors

def clean_data(data: pd.DataFrame, strategy: str='model', file = None, corr_threshold=None, verbose=False) -> pd.DataFrame:
    """Clean dataset of INF- and NAN-values.
    remove constant columns
    
    Keyword arguments:
    dat -- dataframe
    strategy -- cleaning strategy, should be in ['ffill', 'bfill', 'pad',]
    Output:
    data -- cleaned dataframe
    """
    if data.empty:
        return data
        
    logger.info('Start to clean the given dataframe...')
    
    #print('#'*30)
    print('#### clean na and inf in the data \n', file = file)
    #print('#'*30)
    
    data = convert_data_comma_and_set_type_float(data, verbose)
    
    print(f'Number of INF- and NAN-values are: (***{(data == np.inf).sum().sum()}***, ***{data.isna().sum().sum()}***) \n', file = file)
    print('Set all numeric feature type to float32 at first && deal with INF \n', file = file)
    
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # TODO add threshold parameter
    print(f'Remove columns with half of NAN-values: {data.columns[(data.isna().sum()/data.shape[0])>=0.5].to_list()} \n', file = file)
    data = data.dropna(axis=1, thresh=data.shape[0] * .5)
    
    print(f'Check constant columns: {data.columns[(data == data.iloc[0]).all()].to_list()} \n', file = file)
    if data.columns[(data == data.iloc[0]).all()] > 0:
        print('There exist columns with constant value, which will be removed from the feature set')
        data = data.loc[:, (data != data.iloc[0]).any()]

    # TODO also add 'bfill', ...
    if data.isna().sum().sum() > 0:
        
        if strategy == 'model':
            imp = IterativeImputer(max_iter=10, random_state=0)
            tmp = imp.fit_transform(data)
        elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
            imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
            tmp = imp.fit_transform(data)
        elif strategy in ['backfill', 'bfill', 'pad', 'ffill']:
            data = data.fillna(method=strategy)
        else:
            logger.warn(f'Unrecognized strategy {strategy}. Use mean instead')
            
        if tmp.shape[1] != data.shape[1]:
            logger.warn(f'Error appeared while fitting. Use constant filling instead')
            tmp = data.fillna(0)
            
#         data = pd.DataFrame(tmp, columns=data.columns, index=data.index)
        
    if corr_threshold:      
        data, _ = extract_large_correlation(data, threshold=corr_threshold, remove=False, verbose=verbose)    
        
    #logger.info('Remove rows with any nan in the end')
    #data = data.dropna(axis=0, how='any')
    logger.info(f'...End with Data cleaning, number of INF- and NAN-values are now: ({(data == np.inf).sum().sum()}, {data.isna().sum().sum()})')
    print(f'End with Data cleaning, number of INF- and NAN-values are now: (***{(data == np.inf).sum().sum()}***, ***{data.isna().sum().sum()}***) \n', file = file)
    #data = data.reset_index(drop=True)
    return data

def combine_dataframes(data: List, strategy: str='merge_nearest_index', date_col: str='date'):
    if strategy == 'merge_nearest_index':
        data = sort_by_sample_rate(data)
        
def sort_by_sample_rate(data: List):
    data_sorted = sorted(data, key=lambda x: x.index.to_series().diff().median(), reverse=True)
    return data_sorted

def convert_column_comma_and_set_type_float(col: pd.Series,) -> pd.Series:
    """
    Konvertiert Kommazahlen einer Spalte in die englische Schreibweise mit Punkt statt Komma.
    """
    col = col.map(lambda x: x.replace('.', '0.0').replace(',', '.') if type(x) != float else x).astype(float)
    return col

def convert_data_comma_and_set_type_float(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Konvertiert Kommazahlen eines Dataframes in die englische Schreibweise mit Punkt statt Komma.
    """
    data = copy.deepcopy(data)
    for i in data.columns:
        try:
            data[i] = convert_column_comma_and_set_type_float(data[i])
        except:
            if verbose:
                print(f'column {i} is not numerical')
            
    return data

def extract_large_correlation(data, threshold=0.9, remove=False, verbose = True):
    # show corr == 1
    repeat_tuple = []
    tmp = data.corr()
    for i in data.columns[2:]:
        for j in data.columns[2:]:
            if tmp.loc[i,j] >= threshold and i != j:
                if verbose:
                    print(i,j, tmp.loc[i,j])
                    
                repeat_tuple.append((i,j))
    
    if len(repeat_tuple) == 0:
        return data, []

    # repeat tuple to repeat list
    repeat_list = []
    tmp = []
    for i in repeat_tuple:
        idx0 = _check_existence(i[0], repeat_list)
        idx1 = _check_existence(i[1], repeat_list)
                
        if idx0 == -1 and idx1 == -1:
            # both not exist
            repeat_list.append(set([i[0], i[1]]))
        
        elif idx0 == -1:
            idx = idx0 if idx0 != -1 else idx1
            
            repeat_list[idx].add(i[0])
            repeat_list[idx].add(i[1])
    
    if verbose:
        print(f'Repeat columns: {repeat_list}')
    
    #logger.info(f'Found {len(repeat_list)} correlated columns.')
    
    if remove:
        # remove corr == 1
        for i in repeat_list:
            for j in list(i)[1:]:
                data = data.drop(j, axis = 1)             
        logger.info(f'Removed {len(repeat_list) - 1} correlated columns.')
                
    return data, repeat_list

def _check_existence(t: str, lis: list):
    """
    lis is list of list, this function return the list the t(arget) in and return the index
    if not exist, return -1
    """
    for i, j in enumerate(lis):
        if t in j:
            return i
    return -1


def convert_datetime_as_index(df, time_col: Union[str, int, Dict], time_format=None):
    if type(time_col) == str:
        df.index  = pd.to_datetime(df[time_col], dayfirst=True, format=time_format)
        df = df.drop([time_col], axis=1)
    elif type(time_col) == dict:
        #df[time_col['date']] = df[time_col['date']].dt.strftime('%d/%m/%Y')
        #print(df[time_col['date']])
        df.index  = pd.to_datetime(df[time_col['date']] + ' ' + df[time_col['time']], dayfirst=True, format=time_format)
        df = df.drop([time_col['date'], time_col['time']], axis=1)
    
    elif type(time_col) == int:
        time_col = df.columns[time_col]
        df.index  = pd.to_datetime(df[time_col], dayfirst=True, format=time_format)
        df = df.drop([time_col], axis=1)
    
    else:
        print('Unrecognized time column type')
        
    return df
