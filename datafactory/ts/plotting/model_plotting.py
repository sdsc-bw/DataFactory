'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# plot package
from sklearn import tree
import graphviz
import numpy as np
import matplotlib.pyplot as plt

from dtreeviz.trees import dtreeviz # remember to load the package
from tqdm import tqdm
from matplotlib.colors import ListedColormap

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# model packages
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

def compute_fig_from_df(model_type, result, metrics):
    if model_type == 'C':
        return compute_fig_from_classification_df(result, metrics)
    elif model_type == 'R':
        return compute_fig_from_regression_df(result, metrics)
    else:
        logger.warn(f'Unrecognized model_type {model_type}, use regression instead')
        return compute_fig_from_regression_df(result, metrics)

def compute_fig_from_classification_df(result, metrics):
    result = result.loc[:, result.columns != 'value']
    mean_result = result.groupby('model').mean().sort_values('test_roc_auc')
    std_result = result.groupby('model').std().loc[mean_result.index]

    # plot
    traces = []

    for i in metrics:
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
    return fig      

def compute_fig_from_regression_df(result, metrics):
    result = result.loc[:, result.columns != 'value']
    mean_result = result.groupby('model').mean().sort_values('test_neg_mean_absolute_error')
    std_result = result.groupby('model').std().sort_values('test_neg_mean_absolute_error').loc[mean_result.index]

    # plot
    traces = []

    for i in metrics:
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
    
    return fig

def plot_decision_tree(dt, dat, dat_y):
    # DOT data
    dot_data = tree.export_graphviz(dt, out_file=None, 
                                  feature_names=dat.columns,  
                                  class_names='target',
                                  filled=True)
    # Draw graph
    graph = graphviz.Source(dot_data, format="svg").source
    #graph.save('./test.png')

    # viz plot
    viz = dtreeviz(dt, dat, dat_y,
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
