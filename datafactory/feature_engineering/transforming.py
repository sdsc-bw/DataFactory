'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .transformations_unary import *
from .transformations_binary import *
from .transformations_multi import *
from .transformations_transform import *


import sys
sys.path.append('../util')
from ..util.constants import logger


def load_opts(transform_type: str) -> Dict[str, Union[UnaryOpt, BinaryOpt]]:
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

OPTS_UNARY = load_opts(transform_type = 'unary')
OPTS_BINARY = load_opts(transform_type = 'binary')
OPTS_MULTI = load_opts(transform_type = 'multi')
OPTS_CLA_SUPERVISED = load_opts(transform_type= 'cla')
OPTS_REG_SUPERVISED = load_opts(transform_type= 'reg')

def apply_unary_transformations_to_series(value: pd.Series) -> pd.DataFrame:
    """Applies all unary transformations to a given series.
    
    Keyword arguments:
    value -- series
    Output:
    dataframe with transformed series as columns, one column per transformation
    """
    values = []
    logger.info(f'Start to apply unary transformtions to series: {value.name}...')
    for key in OPTS_UNARY.keys():
        logger.info(f'Applying transformation: {OPTS_UNARY[key].name}')
        tmp_value = OPTS_UNARY[key].fit(value)
        values.append(tmp_value)
    logger.info(f'...End with transformation')
    transformed_values = pd.concat(values, axis = 1)
    return transformed_values

def apply_binary_transformations_to_series( value1: pd.Series, value2: pd.Series) -> pd.DataFrame:
    """Applies all binary transformations to a given series.
    
    Keyword arguments:
    value -- series
    Output:
    dataframe with transformed series as columns, one column per transformation
    """
    values = []
    logger.info(f'Start to apply binary transformtions to series {value1.name} and series {value2.name}...')
    for key in OPTS_BINARY.keys():
        logger.info(f'Applying transformation: {OPTS_BINARY[key].name}')
        tmp_value = OPTS_BINARY[key].fit(value1, value2)
        values.append(tmp_value)
    logger.info(f'...End with transformation')
    transformed_values = pd.concat(values, axis = 1)
    return transformed_values
            
def apply_multiple_transformations_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all multiple transformations to a given dataframe.
    
    Keyword arguments:
    df -- dataframe
    Output:
    dataframe with transformed dataframe as columns, one column per transformation
    """
    dfs = []
    logger.info(f'Start to apply multi transformtions to dataframe...')
    for key in OPTS_MULTI.keys():
        logger.info(f'Applying transformation: {OPTS_MULTI[key].name}')
        tmp_df = OPTS_MULTI[key].fit(df)
        dfs.append(tmp_df)
    logger.info(f'...End with transformation')
    transformed_dfs = pd.concat(dfs, axis = 1)
    return transformed_dfs
    
def apply_supervised_transformations_to_dataframe(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                   y_train: pd.Series, y_test: pd.Series, art = 'C') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies all multiple transformations to given test and trainings dataframe including their targets.
    
    Keyword arguments:
    X_train -- training data
    X_test -- test data
    y_train -- training targets
    y_test -- test targets
    Output:
    dataframe with transformed training data as columns, one column per transformation
    and dataframe with transformed test as columns, one column per transformation
    """
    dfs_train, dfs_test = [], []
    logger.info(f'Start to apply supervised transformtions to dataframe...')
    opts = OPTS_CLA_SUPERVISED if art == 'C' else OPTS_REG_SUPERVISED
    for key in opts.keys():
        logger.info(f'Applying transformation: {opts[key].name}')
        tmp_train, tmp_test = opts[key].fit(X_train, X_test, y_train, y_test)
        dfs_train.append(tmp_train)
        dfs_test.append(tmp_test)
    logger.info(f'...End with transformation')
    transformed_dfs_train = pd.concat(dfs_train, axis = 1)
    transformed_dfs_test = pd.concat(dfs_test, axis = 1)
    return transformed_dfs_train, transformed_dfs_test