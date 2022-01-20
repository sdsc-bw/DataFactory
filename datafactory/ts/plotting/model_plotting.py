'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

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