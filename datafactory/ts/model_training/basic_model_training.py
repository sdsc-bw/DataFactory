'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

import pandas as pd
import sys
import time
from tqdm import tqdm
from typing import cast, Any, Dict, List, Tuple, Optional, Union


sys.path.append('../../util')
from ...util.constants import logger
from ...util.models import _get_model

def compare_models(X: pd.DataFrame, y: pd.Series, models: list, model_type: str='C', scoring: Union[str, List]='f1', average: str='micro', cv: int=5):
    # TODO allow multiple metrics
    if type(scoring) != list:
        scoring = [scoring]
    
    # concat scoring and average
    for i in range(len(scoring)):
        if scoring[i] != 'accuracy' and scoring[i] != 'mse' and scoring[i] != 'mae' and scoring[i] != 'explained_variance' and scoring[i] != 'r2':
            scoring[i] = scoring[i] + '_' + average
            
    results = pd.DataFrame(columns = ['model'] + list(map(lambda x: 'test_' + x, scoring)))
    
    counter = 0
    for m in tqdm(models):
        
        model = _get_model(m, X, y, model_type, average)
        
        logger.info("Running cross validation for: " + model.get_name() + "...")

        scores = model.cross_val_score(cv=cv, scoring=scoring)
        for i in range(cv):
            results.loc[counter, 'model'] = model.get_name()  
            for j in scoring:
                results.loc[counter, 'test_' + j] = scores[j][i]
            counter += 1
                               
    return results

