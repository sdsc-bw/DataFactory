'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, SelectPercentile, GenericUnivariateSelect, RFE, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.decomposition import FactorAnalysis
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import sys

def add_time_as_columns(data):
    data['Month'] = data.index.month
    data['Day of the Week'] = data.index.dayofweek
    data['Day'] = data.index.day
    data['Hour'] = data.index.hour
    data['Minute'] = data.index.minute
    data['Second'] = data.index.second
    return data

def feature_selection(dat_x: pd.DataFrame, dat_y: pd.Series, method: str ='select_best', **kw) -> np.array:
    """
    method parameter:
    select_best:
        score_func, regression [f_regression, mutual_info_regression], classification [chi2, f_classif, mutual_info_classif,]
        k,
    select_percentile:
        score_func,
        percentileint
    select_generic:
        score_func,
        mode, {‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’}
        param, int/float depends on mode
    rfe:
        estimator: models include coef_, feature_importances_ features
        n_features_to_select, int/float  larger then 1 -> number of feature; smaller then 1 -> percentile
        step, feature to remove in each step
        importance_getter, str default 'auto'
    rfecv: combine rfe and cv
        estimator
        step
        min_feature_to_select
        cv
        score
        importance_getter
    select_model:
        estimator: 
        threshold: value to use for feature selection
        prefit: bool
        norm_order: 
        max_features:
        importance_getter:
    """
    if method == 'select_best':
        fs = SelectKBest(**kw)
    
    elif method == 'select_percentile':
        fs = SelectPercentile(**kw)
    
    elif method == 'select_generic':
        fs = GenericUnivariateSelect(**kw)
    
    elif method == 'rfe':
        fs = RFE(**kw)
    
    elif method == 'rfecv':
        fs = RFECF(**kw)
    
    elif method == 'select_model':
        fs = SelectFromModel(**kw)
        
    elif method == 'factor_analysis':
        fs = FactorAnalysis(**kw)
    
    else:
        print(f'Unrecognized method {method}, use select_best instead')
        fs = SelectKBest(**kw)
    
    return fs.fit_transform(dat_x, dat_y)