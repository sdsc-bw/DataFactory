'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
import sqlite3

from sklearn.metrics import f1_score
import sklearn.utils

from typing import cast, Any, Dict, List, Tuple, Optional, Union
from tqdm import tqdm

sys.path.append('../feature_engineering')
from ..feature_engineering.transforming import apply_transforms

sys.path.append('../preprocessing')
from ..preprocessing.encoding import categorical_feature_encoding
from ..preprocessing.cleaning import convert_datetime_as_index, convert_data_comma_and_set_type_float

sys.path.append('../../util')
from ...util.constants import logger, available_datatypes
from ...util.metrics import evaluate

def load_dataset(datatype, file_path, is_file=True, time_col=None, time_format=None, sampling_rate=timedelta(days=1), header: str='infer', sep: str=',', index_col: Union[str, int]=0, query: Union[list,str]="""select *""", shuffle: bool=False, agg: str=None, index_start=None, index_end=None, output_path=None, pref=None):
    if datatype == 'sqlite3':
        dfs = load_datasets_from_database(file_path, is_file, query=query)
    else:
        dfs = load_datasets_from_file(datatype, file_path, is_file, header=header, sep=sep, index_col=index_col, time_col=time_col, 
                                      time_format=time_format)
        
    # combine if multiple input files    
    if len(dfs) > 1:    
        df, df_info = combine_df(dfs, sampling_rate, time_col, agg=agg, index_start=index_start, index_end=index_start, 
                        pref=pref)
    else:
        df = list(dfs.values())[0]
        if not agg is None:
            df = categorical_feature_encoding(df)
    
    # shuffle if wanted
    if shuffle:
        df = sklearn.utils.shuffle(df)
        df.reset_index(drop=True, inplace=True)
    
    if output_path:
        df.to_csv(output_path + '/datasets/dataset_unprocessed.csv', sep=sep)
    
    return df
        
def load_datasets_from_file(data_type, path_or_root_dir, is_file, header: str='infer', sep: str=',', index_col: Union[str, int]=0, time_col: str=None, time_format: str=None):
    # TODO check automatically, if is file and which datatype
    dfs = {}
    directory = os.getcwd()
    
    if type(path_or_root_dir) == str:
        path_or_root_dir = [path_or_root_dir]
   
    i = 1
        
    for j in path_or_root_dir:
        if is_file:
            df = _read_file(data_type, j, header=header, sep=sep, index_col=index_col)

            if time_col:
                df = convert_datetime_as_index(df, time_col, time_format=time_format)

            df = convert_data_comma_and_set_type_float(df, verbose=False)

            dfs[j] = df
            i += 1
                
        else:
            fns = os.listdir(j)

            for fn in tqdm(fns):

                df = _read_file(data_type, j + fn, header=header, sep=sep, index_col=index_col)

                if time_col:
                    df = convert_datetime_as_index(df, time_col, time_format=time_format)

                df = convert_data_comma_and_set_type_float(df, verbose=False)

                dfs[fn] = df
                i += 1
    return dfs    



def load_datasets_from_database(database: Union[list,str], query: Union[list,str]="""select *""", shuffle: bool=True):
    """Loads one or multiple datasets from the given databases.
        
    Keyword arguments:
    databases -- sql database
    sep -- seperator of the dataset to seperate cells if csv-file
    shuffle -- if data should be shuffled
    Output:
    dataframe or list of dataframes
    """
    # TODO rework
    if type(database) == str:
        connection = sqlite3.connect(database)
        cursor = connection.cursor()
        query_results = cursor.execute(query).fetchall()
        df = pd.DataFrame(query_results)
        return df
    elif type(database) == list:
        dfs = []
        for i in database:
            connection = sqlite3.connect(i)
            cursor = connection.cursor()
            query_results = cursor.execute(query).fetchall()
            df = pd.DataFrame(query_results)

            if shuffle:
                df = sklearn.utils.shuffle(df)
                df.reset_index(drop=True, inplace=True)
            dfs.append(df)
        return dfs
    else:
        raise AttributeError(f'Unknown datatype of database: {database}')
        

def _read_file(data_type: str, file_path_or_buffer: str, header: str='infer', sep: str=',', index_col: Union[int, str]=0, time_format=None):
    if data_type == 'csv':
        df = pd.read_csv(file_path_or_buffer, sep=sep, header=header)
    elif data_type == 'xml':
        df = pd.read_xml(file_path_or_buffer)
    elif data_type == 'txt':
        df = pd.read_fwf(file_path_or_buffer, header=header)
        
    else:
        raise AttributeError(f'Unknown datatype of file: {data_type}')
  
    if index_col:
        df = df.set_index(index_col)
        
    return df
 
def resample_time_series(data, index_start, index_end, sampling_rate, time_col, agg=None):
    """
    resample the give time series data, when agg == mean, object columns will be removed after the resampling
    Attention:
        one condition of using this method is the index of the given dataframe should be type of datetime
    ============
    Inputs:
    ============
    data, type of DataFrame, the target dataframe that contain the time series data. each column represent a channel,
        and each row is an record. The give
    index_start, type of pd.timeStamp
    index_end, type of pd.timeStamp
    sampling_rate, type of pd.timeStamp
    agg, type of string, one of ['mean', 'max', 'min', 'std'] so far. when agg == mean, only numerical column will be kept after the resampling
    """
    if agg is None:
        return data
    
    data_slice = pd.DataFrame(index = np.arange(index_start, index_end, sampling_rate))
    
    # Feature encoding does something strange, adds indexes
    if not agg is None:
        data = categorical_feature_encoding(data)
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    cols = data.select_dtypes(include=numerics).columns
    data = data[cols]
        
    tmp = []
    cols = data.columns
    
    for i in data_slice.index:
        if agg == 'mean':
            tmp.append(data.loc[i: i + sampling_rate].mean().values)    
        elif agg == 'max':
            tmp.append(data.loc[i: i + sampling_rate].max().values)
        elif agg == 'min':
            tmp.append(data.loc[i: i + sampling_rate].min().values)           
        elif agg == 'std':
            tmp.append(data.loc[i: i + sampling_rate].std().values)           
        else:
            print('Unrecognized agg function, use mean instead')
            tmp.append(data.loc[i: i + sampling_rate].mean().values)


    out = pd.DataFrame(tmp, index = data_slice.index)
    
    out.columns = cols
    
    return out    
    
def combine_df(dfs, sampling_rate, time_col, sep=',', agg: str='mean', index_start=None, index_end=None, pref='S'):
    df_info = pd.DataFrame([], columns=['Dateien', 'Abkürzung', '#Zeilen', '#Spalten', 'Anfangsdatum', 'Enddatum', 
                                        'Mittlere Samplingrate'])
    dfs_resampled = []
    i = 1
    
    for key in tqdm(dfs):
         
        df = dfs[key]    
            
        if pref is None:
            prefix = '-'
        else:
            prefix = pref + str(i)
        filename = key
        
        # set start and end index
        if index_start is None:
            index_start = df.index[0]
        if index_end is None:
            index_end = df.index[-1]
            
        
        # fill info df
        df_info.loc[len(df_info.index)] = [filename, prefix, df.shape[0], df.shape[1], df.index[0], 
                                           df.index[-1], str(df.index.to_series().diff().median())]
        
        df = resample_time_series(df, index_start, index_end, sampling_rate, time_col, agg=agg)
        
        if not pref is None: 
            df = df.add_prefix(prefix  + '_')
        
        dfs_resampled.append(df)
        
        i += 1
    
    df_combined = pd.concat(dfs_resampled, axis=1)
        
    return df_combined, df_info  