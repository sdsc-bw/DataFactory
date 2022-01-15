from sklearn.metrics import f1_score

from .constants import logger

def relative_absolute_error(pred, y):
    dis = abs((pred-y)).sum()
    dis2 = abs((y.mean() - y)).sum()
    if dis2 == 0 :
        return 1
    return dis/dis2
    
def get_score(y_pred, y_test, mtype='C'):
    if mtype == 'C':
        score = f1_score(y_test, y_pred, average='weighted')
    elif mtype == 'R':
        score = 1 - relative_absolute_error(y_pred, y_test)
    else:
        logger.error('Unknown type of model')