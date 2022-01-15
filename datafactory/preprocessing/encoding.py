import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys

sys.path.append('../util')
from ..util.constants import logger

def categorical_feature_encoding(X: pd.DataFrame, y: pd.Series=None, k_term: bool=True):
    """Categorical feature encoding of given dataframes
        
    Keyword arguments:
    X -- data
    y -- labels
    k_term -- whether k-terms should be added as columns
    Output:
    encoded data
    and encoded labels (optional)
    """
    logger.info('Start to transform the categorical columns...')
    # replace original indeices with default ones
    if y is not None:
        out_y = y
        if y.dtype == 'O':
            logger.info('Start with label encoding of the target...')
            out_y = pd.Series(LabelEncoder().fit_transform(y), index = y.index)
            logger.info('...End with Target encoding')
    dat_categ = X.select_dtypes(include=['object'])
    dat_numeric = X.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    dat_kterm = pd.DataFrame(index = X.index)
    # get kterm of categ features
    if k_term:
        logger.info('K-term function is activated, try to extract the k-term for each object columns')
        for i in dat_categ.columns:
            # get k-term feature
            tmp = X[i].value_counts()
            dat_kterm[i + '_kterm'] = X[i].map(lambda x: tmp[x] if x in tmp.index else 0)
        logger.info('...End with k-term feature extraction')
    # onehot encoding and label encoding
    dat_categ_onehot = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values < 8]
    dat_categ_label = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values >= 8]
    flag_onehot = False
    flag_label = False
    oe = OneHotEncoder(drop='first')
    # oe
    if dat_categ_onehot.shape[1] > 0:
        print(type(dat_categ_onehot.columns.to_list()))
        logger.info('Start with one-hot encoding of the following categoric features: '+ (str(dat_categ_onehot.columns.to_list())) + '...')
        dat_onehot = pd.DataFrame(oe.fit_transform(dat_categ_onehot.astype(str)).toarray(), 
                                  columns=oe.get_feature_names(dat_categ_onehot.columns))
        logger.info('...End with one-hot encoding')
        flag_onehot = True
    else:
        dat_onehot = None
    # le
    if dat_categ_label.shape[1] > 0:
        logger.info('Start label encoding of the following categoric features: %s...' %(str(dat_categ_label.columns.to_list())))
        dat_categ_label = dat_categ_label.fillna('NULL')
        dat_label = pd.DataFrame(columns=dat_categ_label.columns)
        for i in dat_categ_label.columns:
            dat_label[i] = LabelEncoder().fit_transform(dat_categ_label[i].astype(str))
        flag_label = True
        logger.info('...End with label encoding')
    else:
        dat_label = None
    # scaling
    # combine
    dat_new = pd.DataFrame()
    if flag_onehot and flag_label:
         dat_new = pd.concat([dat_numeric, dat_onehot, dat_label], axis=1)
    elif flag_onehot:
         dat_new = pd.concat([dat_numeric, dat_onehot], axis=1)
    elif flag_label:
        dat_new = pd.concat([dat_numeric, dat_label], axis=1)
    else:
        dat_new = dat_numeric
    if k_term:
        dat_new = pd.concat([dat_new, dat_kterm], axis = 1)
    logger.info('...End with categorical feature transformation')
    if y is not None:
        return dat_new, out_y
    else:
        return dat_new
    
def date_encoding(value: pd.Series) -> pd.DataFrame:
    logger.info(f'Start to extract datetime information from: {value.name}...')
    tmp = pd.to_datetime(value)
    out = tmp.apply(lambda x: pd.Series([x.year, x.month, x.day, x.dayofyear, x.dayofweek,
                                         x.hour, x.minute, x.second, x.microsecond], 
                                        index = [value.name +'_'+ i for i in ['year', 'month', 'day', 'dayofyear', 'dayofweek', 'hour',
                                                                              'minute', 'second','microsecond']]))
    logger.info('...End with date time information extraction')
    return out