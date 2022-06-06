'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''
import copy
import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from sklearn.metrics import f1_score
import sklearn.utils
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from datetime import datetime
import sys
sys.path.append('../feature_engineering')
from ..feature_engineering.transforming import apply_transforms

sys.path.append('../../util')
from ...util.constants import logger
from ...util.metrics import evaluate

def split_df(df: pd.DataFrame, target_col=None):
    """Splits a dataframe into data and target.
        
    Keyword arguments:
    df -- dataframe
    target_col -- name of target column, if None uses last column
    Output:
    data
    and targets
    """
    if target_col:
        df = df.copy() # WARNING could be problematic for large data
        y = df[target_col]
        X = df.drop([target_col])
    else:
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y
    

def load_dataset_from_dir(path: str, header: str = 'infer', sep: str = ',', index_col: Union[str, int] = 0, time_format: str = None, dayfirst = True):
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
        
def load_dataset_from_file(data_type: str, file_path_or_buffer: Union[list,str], header: str='infer', sep: str=',', index_col: Union[str, int]=0, time_format: str=None, shuffle: bool=True, transform: List=None) -> Union[pd.DataFrame, List]:    
    """Loads one or multiple datasets from the given file paths.
        
    Keyword arguments:
    data_type -- type of the file, should be in ['csv', 'xml', 'txt']
    file_path_or_buffer -- path to dataset, valid xml string or url
    sep -- seperator of the dataset to seperate cells if csv-file
    shuffle -- if data should be shuffled
    Output:
    dataframe or list of dataframes
    """
    
    #TODO maybe method to automatically identify data_type
    if type(file_path_or_buffer) == str:
        df = _read_file(data_type, file_path_or_buffer, header=header, sep=sep, index_col=index_col)

        if transform:
            df = apply_transforms(df, transform)

        if shuffle:
            df = sklearn.utils.shuffle(df)
            df.reset_index(drop=True, inplace=True)
            
        return df
    
    elif type(file_path_or_buffer) == list:
        dfs = []
        for i in file_path_or_buffer:
            df = _read_file(data_type, i, header=header, sep=sep, index_col=index_col)

            if transform:
                df = apply_transforms(df, transform)

            if shuffle:
                df = sklearn.utils.shuffle(df)
                df.reset_index(drop=True, inplace=True)
                
            dfs.append(df)
        return dfs
    else:
        raise AttributeError(f'Unknown datatype of file: {data_type}')

def load_dataset_from_database(database: Union[list,str], query: Union[list,str]="""select *""", shuffle: bool=True, transform: List=None):
    """Loads one or multiple datasets from the given databases.
        
    Keyword arguments:
    databases -- sql database
    sep -- seperator of the dataset to seperate cells if csv-file
    shuffle -- if data should be shuffled
    Output:
    dataframe or list of dataframes
    """
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
            
            if transform:
                df = apply_transforms(df, transform)

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
        
    return df
    
    