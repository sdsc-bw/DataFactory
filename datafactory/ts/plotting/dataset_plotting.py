'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import cast, Any, Dict, List, Tuple, Optional, Union
import os

def plot_linear_for_columns_in_df(df: pd.DataFrame, step: int, cols: List, save_path: str=None, id: int=None):
    """Creates a linear plot of the given columns in the dataframe. Saves the plot at a given path.
        
    Keyword arguments:
    df -- dataframe
    step -- plot the value every ||step|| items
    cols -- columns that should be plotted
    save_path -- path where to save the plot
    id -- ID of the plot
    """
    plt.figure(figsize=(20,6))
    df = df.reset_index(drop = True)
    for i in cols:
        tmp = df.loc[np.arange(0, len(df), step), i]
        plt.plot(tmp.index, tmp, label = i) # /1000
    plt.xlabel('Second')
    plt.legend()
    plt.title(str(id))
    plt.tick_params(axis='x',labelsize=18)
    plt.tick_params(axis='y',labelsize=18)
    plt.legend(prop={'size': 16})
    if save_path:
        save_path = save_path+'_step_'+str(step)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + '/' + '_'.join(cols) + '_' + str(id), transparent=True)
        
def plot_density_for_each_column_in_df(df: pd.DataFrame, save_path: str=None, id=None):
    """Creates a density plots of each column in the dataframe. Saves plot at given path.
        
    Keyword arguments:
    df -- dataframe
    save_path -- path where to save the plot
    id -- ID of the plot
    """
    for i in df.columns:
        plt.figure(figsize=(20,6))
        sns.kdeplot(df[i])
        #plt.xlabel('Second')
        plt.legend()
        plt.title(str(i))
        plt.tick_params(axis='x',labelsize=18)
        plt.tick_params(axis='y',labelsize=18)
        plt.legend(prop={'size': 16})
    if save_path:
        save_path = save_path+'_step_'+str(step)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path + '/' + '_'.join(cols) + '_' + str(id), transparent=True)