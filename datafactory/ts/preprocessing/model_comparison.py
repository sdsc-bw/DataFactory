'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import numpy as np
import pandas as pd

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

def basic_model_comparison_classification(dat: pd.DataFrame, dat_y: pd.Series, models: list):
  """
  run selected models and return dataframe and comparison figure as result
  """
  # setting:
  classifiers = [get_model_with_name_classification(i) for i in models]

  #cla_metrics = ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'f1_micro', 'f1_macro', 'f1_weighted', 'roc_auc']
  cla_metrics = ['accuracy', 'average_precision', 'f1_weighted', 'roc_auc']

  # classification
  counter = 0
  results = pd.DataFrame(columns = ['model', 'index', 'fit_time', 'test_accuracy', 'test_average_precision', 'test_f1_weighted', 'test_roc_auc'])

  for classifier in tqdm(classifiers):
    # train model
    out = cross_validate(classifier, dat, dat_y, scoring = cla_metrics, return_estimator= True)

    # record result
    for i in range(5):
      for j in cla_metrics:
        if str(out['estimator'][0]) == 'DummyClassifier()':
          results.loc[counter, 'model']='baseline'
        else:
          results.loc[counter, 'model'] = str(out['estimator'][0])
        results.loc[counter, 'index'] = i
        results.loc[counter, 'fit_time'] = out['fit_time'].mean()
        results.loc[counter, 'test_'+j] = out['test_'+j][i]
      counter += 1
    
  results = pd.concat([results.iloc[:,0], results.iloc[:, 1:].astype(float)], axis = 1)

  mean_result = results.groupby('model').mean().sort_values('test_roc_auc')
  std_result = results.groupby('model').std().loc[mean_result.index]

  # plot
  traces = []

  for i in cla_metrics:
    traces.append(go.Bar(
        x = mean_result.index,#['model'],
        y = mean_result['test_'+i],
        error_y= dict(
        type= 'data',
        array= std_result['test_'+i],
        visible= True
        ),
        name = i,
        ))
    
  fig = go.Figure(traces)

  return results, fig        
        
def basic_model_comparison_regression(dat: pd.DataFrame, dat_y: pd.Series, models: list):
  # setting:
  regressors = [get_model_with_name_regressor(i) for i in models]
  reg_metrics = ['explained_variance', 'max_error', 'neg_mean_absolute_error','neg_mean_squared_error','r2']

  # classification
  counter = 0
  results = pd.DataFrame(columns = ['model', 'index', 'fit_time', 'test_explained_variance', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_r2'])

  for regressor in tqdm(regressors):
    # train model
    out = cross_validate(regressor, dat, dat_y, scoring = reg_metrics, return_estimator= True)

    # record result
    for i in range(5):
      for j in reg_metrics:
        if str(out['estimator'][0]) == 'DummyRegressor()':
          results.loc[counter, 'model']='baseline'
        else:
          results.loc[counter, 'model'] = str(out['estimator'][0])
        results.loc[counter, 'index'] = i
        results.loc[counter, 'fit_time'] = out['fit_time'].mean()
        results.loc[counter, 'test_'+j] = out['test_'+j][i]
      counter += 1

  results = pd.concat([results.iloc[:,0], results.iloc[:, 1:].astype(float)], axis = 1)

  mean_result = results.groupby('model').mean().sort_values('test_neg_mean_absolute_error')
  std_result = results.groupby('model').std().sort_values('test_neg_mean_absolute_error').loc[mean_result.index]

  # plot
  traces = []

  for i in reg_metrics:
    traces.append(go.Bar(
        x = mean_result.index,#['model'],
        y = mean_result['test_'+i],
        error_y= dict(
        type= 'data',
        array= std_result['test_'+i],
        visible= True
        ),
        name = i,
        ))
    
  fig = go.Figure(traces)

  return results, fig   

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
