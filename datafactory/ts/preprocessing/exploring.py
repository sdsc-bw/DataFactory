'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''
import time

import pandas as pd
import numpy as np

import sys
sys.path.append('../../util')
from ...util.constants import logger

def get_feature_info(dfs, locations=None):

    feature_info = dfs.isnull().sum()

    statistical_info = dfs.describe().T

    feature_info = pd.concat([feature_info, statistical_info], axis=1)
    feature_info.insert(0, 'features', statistical_info.index)
        
    return feature_info

def get_correlations(dfs):
    corrs_table = {}
    corrs_fig = {}
    for df_keys in dfs:
        corrs_table[df_keys] = dfs[df_keys].corr()
        corrs_fig[df_keys] = px.imshow(corrs_table[df_keys])
    return corrs_table, corrs_fig
