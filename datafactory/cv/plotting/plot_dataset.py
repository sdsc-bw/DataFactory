'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import numpy as np
import matplotlib.pyplot as plt

def plot_images_as_grid(X, y, n_images: int=10, n_cols: int=5, color: str='rgb'):
    # TODO add rgb
    if color == 'gray':
        cmap = plt.cm.gray_r
    else:
        cmap = None
        
    n_rows =  n_images // n_cols
    n_rows += n_images % n_cols    
    position = range(1, n_images + 1)
    
    fig = plt.figure(1)
    for i in range(n_images):
        ax = fig.add_subplot(n_rows, n_cols, position[i])
        ax.set_axis_off()
        image = X[i]
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        ax.imshow(image, cmap=cmap)
        ax.set_title("Class: %i" % y[i])