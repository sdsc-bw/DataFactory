'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''
import copy
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
    elif datatype == 'directory':
        dfs = load_datasets_from_dir(file_path, is_file, query=query)
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

def load_datasets_from_dir(path: str, header: str = 'infer', sep: str = ',', index_col: Union[str, int] = 0, time_format: str = None, dayfirst = True):
    """
    Attention: timestamp should be set to index
    Keyword arguments:
    data_type -- type of the file, should be in ['csv', 'xml', 'txt']
    file_path_or_buffer -- path to dataset, valid xml string or url
    sep -- seperator of the dataset to seperate cells if csv-file
    index_col -- name or index of the index_column, should the type of datetime
    """
    ## load data
    filenames = [os.path.join(path, i) for i in os.listdir(path) if i.split('.')[-1] == 'csv']
    
    dfs = []
    for ind, filename in enumerate(filenames):
        if type(index_col) is list:
            # load data
            tmp = pd.read_csv(filename, header = 'infer',sep=sep)
            tmp.index = tmp.apply(lambda x: ' '.join(x[index_col]), axis = 1) # attention, only work if index item is type of object
            for col in index_col:
                del tmp[col]
        
        else:
            tmp = pd.read_csv(filename, sep=sep, header='infer', index_col = index_col)
            
        # convert column to numeric
        tmp = _convert_df_comma_and_set_type_float(tmp)

        # select only numeric column
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols = tmp.select_dtypes(include=numerics).columns
        tmp = tmp[cols]

        # add pref 
        tmp.columns = [str(ind) + '_'+ i for i in tmp.columns]

        dfs.append(tmp)
            
        
    ## transform index type to datetime
    index_fails = []
    for i, df in enumerate(dfs):
        try:
            df.index = pd.to_datetime(df.index, format=time_format)
        except:
            print(f'Warning: Can not convert index of the {i} dataframe to datetime, please check the data again')
            index_fails.append(i)
            
    if len(index_fails) == len(df):
        # no file contain a meaningful datetime
        print('None of the given files contain a meaningful datetime, generate new timestamp automatically. remember that this timestamp means nothing!!!, it will also affect the table combination later.')
        
        date_start = datetime.strptime("90/01/01 00:00", "%d/%m/%y %H:%M")
        for df in dfs:
            date_end = date_start + timedelta(hours = len(df))
            grad = timedelta(hours = 1)
            df.index = np.arange(date_start, date_end, grad)
            
    elif len(index_fails) == 0:
        pass
    
    else:
        # index of some of the files can not be convert to datetime
        print('Some of the given files contain a illegal datetime, remove this datasets from the list')
        tmp = []
        [tmp.append(dfs[i])for i in index_fails]
        dfs = tmp
        del tmp
    
    ## combine multi-dataset, if necessary: list -> pd.DataFrame
    if len(dfs) > 1:
        # get minimal distance and datetime, get maximal datetime, recreate dataframe and load data and combine
        li_date_start = []
        li_date_end = []
        li_date_gap = []
        
        for df in dfs:
            li_date_start.append(df.index[0])
            li_date_end.append(df.index[-1])
            li_date_gap.append(df.index[1] - df.index[0]) ## ATTENTION!!!!!!!: this method is naive, replace it later.
        
        date_start = min(li_date_start)
        date_end = min(li_date_end)
        date_gap = min(li_date_gap)
        cdf = pd.DataFrame(index = np.arange(date_start, date_end, date_gap))
        
        # combine data
        df_new = []
        for df in dfs:
            tmp_df = []
            for i in cdf.index:
                tmp_df.append(df.iloc[i: i + grad].mean().values) # mean to ensure if smallest gap is not the smallest one
                
            df_new.append(pd.DataFrame(tmp_df, index = cdf.index))
        
        df_new = pd.concat(df_new, axis = 1)
    
    else:
        df = dfs[0]
    
    return df

def _convert_column_comma_and_set_type_float(col: pd.Series,) -> pd.Series:
    return col.map(lambda x: x.replace(',', '.') if type(x) != float else x).astype(float)

def _convert_df_comma_and_set_type_float(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    It's useful when in the given dataset the point of number is set to comma. This method can turn it back
    """
    df = copy.deepcopy(df)
    for i in df.columns:
        try:
            df[i] = _convert_column_comma_and_set_type_float(df[i])
        except:
            if verbose:
                print(f'column {i} is not numerical')
            
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



def load_datasets_from_database(database: Union[list,str], query: Union[list,str]="""select *"""):
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
        

def _read_file(data_type: str, file_path_or_buffer: str, header: str='infer', sep: str=',', index_col: Union[int, str]=0):
    if data_type == 'csv':
        df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, index_col = index_col)
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