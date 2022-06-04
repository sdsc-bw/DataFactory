'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''
import time

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import sys
sys.path.append('../../util')
from ...util.constants import logger

def compute_feature_importance_of_random_forest(X: pd.DataFrame, y: pd.Series, model_type: str = 'C', strategy: str = 'mdi', file = None):
    """
    ================
    Parameters:
    ================
    X, type of pd.DataFrame
    y, type of pd.Series
    model_type, type of str
        the model_type of the task, classification or regression
    strategy: type of str
        the strategy used to calculate the feature importance. Two strategies are available [mdi (mean decrease in impurity), permutation]
        default to use mdi
    file, type of file
        the target file to save the output

    ================
    Output:
    ================
    fig， type of plotly object
        the bar plot showing the feature importance
    forest_importances, type of pd.Series
        the index is the feature names and the value is the corresponding importance of the feature
    """
    if model_type == 'C':
        forest = RandomForestClassifier(random_state=0)

    elif model_type == 'R':
        forest = RandomForestRegressor(random_state=0)

    else:
        logger.warn('Unrecognized model_type of task, use regression instead')
        forest = RandomForestRegressor(random_state=0)

    forest.fit(X, y)

    # extract feature importance
    if strategy == 'mdi':
        start_time = time.time()
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        elapsed_time = time.time() - start_time

    elif strategy == 'permutation':
        start_time = time.time()
        result = permutation_importance(
            forest, dat, dat_y, n_repeats=10, random_state=42, n_jobs=2
        )
        importances = result.importances_mean

        elapsed_time = time.time() - start_time

    else:
        logger.warn('Unrecognized given strategy, use mdi instead ')

        start_time = time.time()
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        elapsed_time = time.time() - start_time

    forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending = False)

    logger.info(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    
    return forest_importances

