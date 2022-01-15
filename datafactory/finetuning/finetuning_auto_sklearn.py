import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('../util')
from ..util.constants import logger
from ..util.metrics import get_score

def finetune_auto_sklearn(X: pd.DataFrame, y: pd.Series=None, mtype: str='C'):
    """Finetunes sklearn models with the library auto-sklearn.
        
    Keyword arguments:
    X -- data
    y -- targets
    mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
    Output:
    the model with the highest score
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    if mtype == 'C':
        automl = autosklearn.classification.AutoSklearnClassifier()
    elif mtype == 'R':
        automl = autosklearn.classification.AutoSklearnRegressor()
    else:
        logger.error('Unknown type of model')
    automl.fit(X_train, y_train)
        
    automl.cv_results_
    automl.performance_over_time_.plot(x='Timestamp', kind='line', legend=True, title='Auto-sklearn accuracy over time',
                                       grid=True)
    plt.show()
    return automl