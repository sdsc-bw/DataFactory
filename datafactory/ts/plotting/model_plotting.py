'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# plo package
from sklearn import tree
import graphviz
import numpy as np
import matplotlib.pyplot as plt

from dtreeviz.trees import dtreeviz # remember to load the package
from tqdm import tqdm
from matplotlib.colors import ListedColormap

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

def plot_decision_tree_classification(dat, dat_y):
  # train decision tree model
  X_train, X_test, y_train, y_test = train_test_split(dat, dat_y, random_state=0)
  clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)

  # DOT data
  dot_data = tree.export_graphviz(clf, out_file=None, 
                                  feature_names=dat.columns,  
                                  class_names='target',
                                  filled=True)
  # Draw graph
  graph = graphviz.Source(dot_data, format="png") 
  #graph.save('./test.png')

  # viz plot
  viz = dtreeviz(clf, dat, dat_y,
                  target_name="target",
                  feature_names=dat.columns,
                  class_names=list('target'))

  return graph, viz       

def plot_model_comparison(self, x_results: list , y_results: list, ptype: str='plot', title: str='', save_path: str=None, id: int=None):
    """Creates a plot of the given columns in the dataframe. Saves the plot at a given path.
        
    Keyword arguments:
    x_results -- x coordinates
    y_results -- y coordinates
    ptype -- type of plot, should be in ['plot', 'bar']
    title -- title of the plot
    save_path -- path where to save the plot
    id -- ID of the plot
    """
    n_figures = len(y_results)
    fig, axs = plt.subplots(n_figures, constrained_layout=True, figsize=(10, 10))
    fig.suptitle(title)
    fig = plt.figure()

    if ptype == 'plot':
        for i in range(len(y_results)):
            if len(x_results) == 1:
                axs[i].plot(x_results[0], y_results[i])
            else:
                axs[i].plot(x_results[i], y_results[i])
    elif ptype == 'bar':
        for i in range(len(x_results)):
            if len(x_results) == 1:
                axs[i].bar(x_results[0], y_results[i])
                axs[i].tick_params(axis='x', labelrotation=45)
                axs[i].set_ylim([0, 1.1])
                axs[i].set_yticks(np.arange(0, 1.5, 0.25))
            else:
                axs[i].bar(x_results[i], y_results[i])
                axs[i].tick_params(axis='x', labelrotation=45)
                axs[i].set_ylim([0, 1.1])
                axs[i].set_yticks(np.arange(0, 1.5, 0.25))
    plt.show()
    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + '/' + '_'.join(cols) + '_' + str(id), transparent=True)

def plot_feature_importance_of_random_forest(X: pd.DataFrame, y: pd.Series, art: str = 'C', strategy: str = 'mdi', file = None):
    """
    ================
    Parameters:
    ================
    X, type of pd.DataFrame
    y, type of pd.Series
    art, type of str
        the art of the task, classification or regression
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
    if art == 'C':
    forest = RandomForestClassifier(random_state=0)

    elif art == 'R':
        forest = RandomForestRegressor(random_state=0)

    else:
        print('Unrecognized art of task, use regression instead')
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
        print('Unrecognized given strategy, use mdi instead ')

        start_time = time.time()
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        elapsed_time = time.time() - start_time

    forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending = False)

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    # plot
    trace1 = go.Bar(
        x = forest_importances.index,
        y = forest_importances.values,
        )

    data = [trace1]
    fig = go.Figure(data = data)
    return fig, forest_importances
