'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from tsai.all import *
from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .constants import logger

def get_loss_fastai(loss:str):
    if loss == 'cross_entropy':
        return CrossEntropyLossFlat() # Classification
    elif loss == 'smooth_cross_entropy':
        return LabelSmoothingCrossEntropyFlat() # Classification
    elif loss == 'l1':
        return L1LossFlat() # Regression/Forecasting
    elif loss == 'focal':
        return FocalLoss() # Classification
    elif loss == 'dice':
        return DiceLoss() # Classification
    elif loss == 'bce':
        return BCEWithLogitsLossFlat() # Regression/Forecasting
    else:
        logger.warn(f'Unknown loss: {loss}. Using default Cross Entropy instead')
        return CrossEntropyLossFlat()
