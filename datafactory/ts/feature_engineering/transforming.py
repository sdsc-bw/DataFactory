'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .transformations_unary import *
from .transformations_binary import *
from .transformations_multi import *
from .transformations_transform import *

import sys
sys.path.append('../../util')
from ...util.constants import logger    
            
def load_transforms(transform_type: str) -> Dict[str, Union[UnaryOpt, BinaryOpt]]:
    """Retruns all transformations of a given type.
    
    Keyword arguments:
    transform_type -- Transformation type
    Output:
    transformations of the given type
    """
    if transform_type == 'unary':
        operators = {'abs': Abs(), 'cos': Cos(), 'degree': Degree(), 'exp': Exp(), 'ln': Ln(), 
                     'negative': Negative(), 'radian': Radian(), 'reciprocal': Reciprocal(), 'sin': Sin(), 
                     'sigmoid': Sigmoid(), 'square': Square(), 'tanh': Tanh(), 'relu': Relu(), 
                     'sqrt': Sqrt(), 'binning': Binning(), 'ktermfreq': KTermFreq()}
    elif transform_type == 'binary':
        operators = {'div': Div(), 'minus': Minus(), 'add': Add(), 'product': Product()}
    elif transform_type == 'multi':
        operators = {'clustering': Clustering(), 'diff': Diff(), 'minmaxnorm': Minmaxnorm(),
                     'winagg': WinAgg(), 'zscore': Zscore(), 'nominalExpansion': NominalExpansion(),
                     'isomap': IsoMap(), 'leakyinfosvr': LeakyInfoSVR(), 'kernelAppRBF': KernelApproxRBF()}
    elif transform_type == 'cla':
        operators = {'dfCla': DecisionTreeClassifierTransform(), 'mlpCla': MLPClassifierTransform(),
                     'knCla': NearestNeighborsClassifierTransform(), 'svCla': SVCTransform(), 
                     'gdwCla': GauDotClassifierTransform(), 'geCla': GauExpClassifierTransform(),
                     'grbfCla': GauRBFClassifierTransform(), 'rfCla': RandomForestClassifierTransform(),
                     'xgbCla': XGBClassifierTransform()}
    elif transform_type == 'reg':
        operators = {'dtReg': DecisionTreeRegressorTransform(), 'liReg': LinearRegressorTransform(),
                     'mlpReg': MLPRegressorTransform(), 'knReg': NearestNeighborsRegressorTransform(),
                     'svReg': SVRTransform(), 'gdwReg': GauDotWhiteRegressorTransform(),
                     'geReg': GauExpRegressorTransform(), 'grbfReg': GauRBFRegressorTransform(),
                     'rfReg': RandomForestRegressorTransform(), 'xgbReg': XGBRegressorTransform()}
    return operators

TRANSFORMS_UNARY = load_transforms(transform_type = 'unary')
TRANSFORMS_BINARY = load_transforms(transform_type = 'binary')
TRANSFORMS_MULTI = load_transforms(transform_type = 'multi')
TRANSFORMS_CLA_SUPERVISED = load_transforms(transform_type= 'cla')
TRANSFORMS_REG_SUPERVISED = load_transforms(transform_type= 'reg')

def apply_transforms(df: pd.DataFrame, transform: List):
    """Applies the defined transformations to the given dataframe.
    
    Keyword arguments:
    df -- dataframe
    transform -- dict of the tranformation as key and the column names where the transformation should be applied 
    Output:
    input dataframe with additional transformed series as columns, one column per transformation
    """
    new_features = []
    for t in transform:
        trfm = t[0]
        if trfm in TRANSFORMS_UNARY:
            value = df[t[1]]
            tmp = apply_unary_transformations_to_series(value, transform=[trfm])
        elif trfm in TRANSFORMS_BINARY:
            value1 =  df[t[1]]
            value2 =  df[t[2]]
            tmp = apply_binary_transformations_to_series(value1, value2, transform=[trfm])
        elif trfm in TRANSFORMS_MULTI:
            tmp_df = df[list(t[1:])]
            tmp = apply_multiple_transformations_to_dataframe(tmp_df, transform=[trfm])
        elif trfm in TRANSFORMS_CLA_SUPERVISED or trfm in TRANSFORMS_REG_SUPERVISED:
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1])
            tmp = apply_supervised_transformations_to_dataframe(X_train, X_test, y_train, y_test, transform=[trfm])
        else:
            logger.error(f'Unknown transformation')
        new_features.append(tmp)
    new_features = pd.concat(new_features, axis = 1)
    new_df = pd.concat([df, new_features], axis = 1)
    return new_df

def apply_unary_transformations_to_series(value: pd.Series, transform: List=None) -> pd.DataFrame:
    """Applies all unary or the defined transformations to a given series.
    
    Keyword arguments:
    value -- series
    transform -- Transformations to be applied
    Output:
    dataframe with transformed series as columns, one column per transformation
    """
    values = []
    logger.info(f'Start to apply unary transformtions to series: {value.name}...')
    if transform:
        for t in transform:
            logger.info(f'Applying transformation: {TRANSFORMS_UNARY[t]}')
            tmp_value = TRANSFORMS_UNARY[t].fit(value)
            values.append(tmp_value)
    else:        
        for key in TRANSFORMS_UNARY.keys():
            logger.info(f'Applying transformation: {TRANSFORMS_UNARY[key].name}')
            tmp_value = TRANSFORMS_UNARY[key].fit(value)
            values.append(tmp_value)
    logger.info(f'...End with unary transformation')
    transformed_values = pd.concat(values, axis = 1)
    return transformed_values

def apply_binary_transformations_to_series(value1: pd.Series, value2: pd.Series, transform: List=None) -> pd.DataFrame:
    """Applies all binary or the defined transformations to a given series.
    
    Keyword arguments:
    value1 -- first series 
    value2 -- second series
    transform -- Transformations to be applied
    Output:
    dataframe with transformed series as columns, one column per transformation
    """
    values = []
    logger.info(f'Start to apply binary transformtions to series {value1.name} and series {value2.name}...')
    if transform:
        for t in transform:
            logger.info(f'Applying transformation: {TRANSFORMS_BINARY[t].name}')
            tmp_value = TRANSFORMS_BINARY[t].fit(value1, value2)
            values.append(tmp_value)
    else:
        for key in TRANSFORMS_BINARY.keys():
            logger.info(f'Applying transformation: {TRANSFORMS_BINARY[key].name}')
            tmp_value = TRANSFORMS_BINARY[key].fit(value1, value2)
            values.append(tmp_value)
    logger.info(f'...End with binary transformation')
    transformed_values = pd.concat(values, axis = 1)
    return transformed_values
            
def apply_multiple_transformations_to_dataframe(df: pd.DataFrame, transform: List=None) -> pd.DataFrame:
    """Applies all or the defined multiple transformations to a given dataframe.
    
    Keyword arguments:
    df -- dataframe
    transform -- Transformations to be applied
    Output:
    dataframe with transformed dataframe as columns, one column per transformation
    """
    dfs = []
    logger.info(f'Start to apply multi transformtions to dataframe...')
    if transform:
        for t in transform:
            logger.info(f'Applying transformation: {TRANSFORMS_MULTI[t].name}')
            tmp_df = TRANSFORMS_MULTI[t].fit(df)
            dfs.append(tmp_df)
    else:
        for key in TRANSFORMS_MULTI.keys():
            logger.info(f'Applying transformation: {TRANSFORMS_MULTI[key].name}')
            tmp_df = TRANSFORMS_MULTI[key].fit(df)
            dfs.append(tmp_df)
    logger.info(f'...End with multi transformation')
    transformed_dfs = pd.concat(dfs, axis = 1)
    return transformed_dfs
    
def apply_supervised_transformations_to_dataframe(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                   y_train: pd.Series, y_test: pd.Series, mtype='C', 
                                                  transform: List=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies transformations for classification to given test and trainings dataframe including their targets.
    
    Keyword arguments:
    X_train -- training data
    X_test -- test data
    y_train -- training targets
    y_test -- test targets
    transform -- Transformations to be applied
    Output:
    dataframe with transformed training data as columns, one column per transformation
    and dataframe with transformed test as columns, one column per transformation
    """
    dfs_train, dfs_test = [], []
    logger.info(f'Start to apply supervised transformtions to dataframe...')
    if transform:
        for t in transform:
            trfm = TRANSFORMS_CLA_SUPERVISED[t] if t in TRANSFORMS_CLA_SUPERVISED else TRANSFORMS_REG_SUPERVISED[t]
            logger.info(f'Applying transformation: {trfm.name}')
            tmp_train, tmp_test = trfm.fit(X_train, X_test, y_train, y_test)
            dfs_train.append(tmp_train)
            dfs_test.append(tmp_test)
    else:
        opts = TRANSFORMS_CLA_SUPERVISED if mtype == 'C' else TRANSFORMS_REG_SUPERVISED
        for key in opts.keys():
            logger.info(f'Applying transformation: {opts[key].name}')
            tmp_train, tmp_test = opts[key].fit(X_train, X_test, y_train, y_test)
            dfs_train.append(tmp_train)
            dfs_test.append(tmp_test)
    logger.info(f'...End with supervised transformation')
    transformed_dfs_train = pd.concat(dfs_train, axis = 1)
    transformed_dfs_test = pd.concat(dfs_test, axis = 1)
    return transformed_dfs_train, transformed_dfs_test