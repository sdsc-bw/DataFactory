'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor 
from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../../util')
from ...util.constants import logger


def basic_model_comparison(dat: pd.DataFrame, dat_y: pd.Series, models: list, metrics, model_type='C'):
    logger.info('Start computing basic models...')
    if model_type == 'C':
        return basic_model_comparison_classification(dat, dat_y, models, metrics)
    elif model_type == 'R':
        return basic_model_comparison_regression(dat, dat_y, models, metrics)
    else:
        logger.warn(f'Unrecognized model_type {model_type}, use regression instead')
        return basic_model_comparison_regression(dat, dat_y, models, metrics)
    logger.info('...End with computing basic models')

def basic_model_comparison_classification(dat: pd.DataFrame, dat_y: pd.Series, models: list, metrics: list):
    """
    run selected models and return dataframe and comparison figure as result
    """
    # setting:
    classifiers = [get_model_with_name_classification(i['value']) for i in models]
    values = [i['value'] for i in models]

    # classification
    counter = 0
    results = pd.DataFrame(columns = ['model', 'index', 'fit_time', 'test_accuracy', 'test_average_precision', 'test_f1_weighted', 'test_roc_auc', 'value'])

    for classifier, value in zip(tqdm(classifiers), values):
        # train model
        out = cross_validate(classifier, dat, dat_y, scoring = metrics, return_estimator= True)

        # record result
        for i in range(5):
            for j in metrics:
                if str(out['estimator'][0]).startswith('DecisionTreeClassifier'):
                    dt = out['estimator'][0]
                if str(out['estimator'][0]) == 'DummyClassifier()':
                    results.loc[counter, 'model']='baseline'
                else:
                    results.loc[counter, 'model'] = str(out['estimator'][0])
                results.loc[counter, 'index'] = i
                results.loc[counter, 'value'] = value
                results.loc[counter, 'fit_time'] = out['fit_time'].mean()
                results.loc[counter, 'test_'+j] = out['test_'+j][i]
            counter += 1
    
    #results = pd.concat([results.iloc[:,0], results.iloc[:, 1:].astype(float)], axis = 1) 

    return results, dt    
        
def basic_model_comparison_regression(dat: pd.DataFrame, dat_y: pd.Series, models: list, metrics: list):
    # setting:
    regressors = [get_model_with_name_regressor(i['value']) for i in models]
    values = [i['value'] for i in models]

    # regression
    counter = 0
    results = pd.DataFrame(columns = ['model', 'index', 'fit_time', 'test_explained_variance', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_r2'])

    for regressor in tqdm(regressors):
        # train model
        out = cross_validate(regressor, dat, dat_y, scoring = metrics, return_estimator= True)

        # record result
        for i in range(5):
            for j in metrics:
                if str(out['estimator'][0]).startswith('DecisionTreeRegressor'):
                    dt = out['estimator'][0]
                if str(out['estimator'][0]) == 'DummyRegressor()':
                    results.loc[counter, 'model']='baseline'
                else:
                    results.loc[counter, 'model'] = str(out['estimator'][0])
                results.loc[counter, 'index'] = i
                results.loc[counter, 'value'] = value
                results.loc[counter, 'fit_time'] = out['fit_time'].mean()
                results.loc[counter, 'test_'+j] = out['test_'+j][i]
            counter += 1

    #results = pd.concat([results.iloc[:,0], results.iloc[:, 1:].astype(float)], axis = 1)

    return results, dt

def get_model_with_name_classification(name:str):
    if name == 'baseline':
        model = DummyClassifier()
    
    elif name == 'knn':
        model = KNeighborsClassifier(3)
    
    elif name == 'svc':
        model = SVC(gamma=2, C=1)
    
    elif name == 'gaussianprocess':
        model = GaussianProcessClassifier(1.0 * RBF(1.0))
    
    elif name == 'decisiontree':
        model = DecisionTreeClassifier(max_depth=5)
    
    elif name == 'randomforest':
        model = RandomForestClassifier()
    
    elif name == 'mlp':
        model = MLPClassifier(max_iter=1000)
    
    elif name == 'adabbost':
        model = AdaBoostClassifier()
    
    elif name == 'gaussian-nb':
        model = GaussianNB()
    
    elif name == 'qda':
        model = QuadraticDiscriminantAnalysis()
    
    else:
        model = RandomForestClassifier()
        
    return model
    
def get_model_with_name_regression(name:str):
    if name == 'baseline':
        model = DummyRegressor()
        
    elif name == 'linear':
        model = LinearRegression()
        
    elif name == 'svr':
        model = SVR()
    
    elif name == 'svr-poly':
        model = SVR(kernel='poly')
    
    elif name == 'svr-sigmoid':
        model = SVR(kernel='sigmoid')
    
    elif name == 'gaussianprocess':
        model = GaussianProcessRegressor()
    
    elif name == 'gaussianprocess-dw':
        model = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())
    
    elif name == 'decisiontree':
        model = DecisionTreeRegressor()
    
    elif name == 'randomforest':
        model = RandomForestRegressor()
    
    elif name == 'mlp':
        model = MLPRegressor(max_iter=1000)
    
    elif name == 'adaboost':
        model = AdaBoostRegressor()
    
    else:
        model = RandomForestRegressor()
    
    return model
