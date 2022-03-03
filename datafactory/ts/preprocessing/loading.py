'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from sklearn.metrics import f1_score
import sklearn.utils
from typing import cast, Any, Dict, List, Tuple, Optional, Union
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
    

def load_dataset_from_file(data_type: str, file_path_or_buffer: Union[list,str], header=None, sep: str=',', shuffle: bool=True, transform: List=None) -> Union[pd.DataFrame, List]:    
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
        df = _read_file(data_type, i, header=header, sep=sep)

        if transform:
            df = apply_transforms(df, transform)

        if shuffle:
            df = sklearn.utils.shuffle(df)
            df.reset_index(drop=True, inplace=True)
            
        return df
    elif type(file_path_or_buffer) == list:
        dfs = []
        for i in file_path_or_buffer:
            df = _read_file(data_type, i, header=header, sep=sep)

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
        

def _read_file(data_type: str, file_path_or_buffer: str, header=None, sep=','):
    if data_type == 'csv':
        df = pd.read_csv(file_path_or_buffer, sep=sep, index_col=0, header=header)
    elif data_type == 'xml':
        df = pd.read_xml(file_path_or_buffer)
    elif data_type == 'txt':
        df = pd.read_fwf(file_path_or_buffer, header=header, index_col=0)
    else:
        raise AttributeError(f'Unknown datatype of file: {data_type}')
    return df
    
    