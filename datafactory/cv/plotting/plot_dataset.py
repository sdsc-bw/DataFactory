'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import cast, Any, Dict, List, Tuple, Optional, Union

def plot_images_as_grid(X, y, n_images: int=10, n_cols: int=5, idx: Dict=None):
    # TODO add rgb
    if X[0].shape[0] == 1:
        cmap = plt.cm.gray_r
    else:
        cmap = None
        
    n_rows =  n_images // n_cols
    n_rows += n_images % n_cols    
    position = range(1, n_images + 1)
    
    fig = plt.figure(figsize=(15, 10))
    for i in range(n_images):
        ax = fig.add_subplot(n_rows, n_cols, position[i])
        ax.set_axis_off()
        image = X[i]
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        elif image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        ax.imshow(image, cmap=cmap)
        if idx:
            ax.set_title("Class: %s" % idx[str(y[i])])
        else:
            ax.set_title("Class: %i" % y[i])