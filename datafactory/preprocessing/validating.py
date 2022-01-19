'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import numpy as np

def valid_col(col: pd.Series) -> bool:
    """Checks if there are inf, NaN, or too many large/small numbers for float32. This column should be discarded.
    
    Keyword arguments:
    col -- column to be checked
    Output:
    shows if column contains invalid values
    """
    if (col.isna().any() or
        (col == np.inf).any() or
        (col > np.finfo(np.float32).max).any() or
        (col < np.finfo(np.float32).min).any() or
        (abs(col - 0.0) < 0.0001).sum() / len(col) > 0.8):
        return False
    return True

# TODO add valid_dataframe