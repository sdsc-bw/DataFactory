import pandas as pd
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .model import SklearnModel

class DecisionTree(SklearnModel):
    
    def __init__(self, X: pd.Series, y: pd.Series, mtype: str, params:Dict=dict()):
        super(DecisionTree, self).__init__(X, y, mtype, params)
        if self.mtype == 'C':
            self.model = DecisionTreeClassifier(**params)
        elif self.mtype == 'R':
            self.model = DecisionTreeRegressor(**params)
        else:
            logger.error('Unknown type of model')
         
        self.name= "Decision Tree"
        self.id = "decision_tree"