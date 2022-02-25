'''
Copyright (c) Smart Data Solution Center Baden-WÃƒÂ¼rttemberg 2021,
All rights reserved.
'''

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys

sys.path.append('../../util')
from ...util.constants import logger, bcolors

def categorical_feature_encoding(X: pd.DataFrame, y: pd.Series=None, k_term: bool=True, file = None):
    """Categorical feature encoding of given dataframes, generate k-term feature and extract date feature
    
    Notice - disadvantage: if NA exist, fill na with new label 'NULL'.
        
    Keyword arguments:
    X -- data
    y -- labels
    k_term -- whether k-terms should be added as columns
    file -- file to save the function information
    Output:
    encoded data
    and encoded labels (optional)
    """
    logger.info('Start to transform the categorical columns...')
    print('#'*30, file = file)
    print('Encoding and categorical features extraction', file = file)
    print('#'*30, file = file)
    # check date data 
    
    tmp_datetime = dt_transform(X)

    if tmp_datetime.shape[1]>1:
        logger.warning('Two date columns exist, please check the given dataset')
        
    if tmp_datetime.shape[1]>0:
        print(f'{bcolors.HEADER}{bcolors.BOLD}{tmp_datetime.shape[1]}{bcolors.ENDC} Date feature(s) detected, try to extract feature from the date feature', file = file)
        # drop date time column in the original dataframe
        X = X.drop(tmp_datetime.columns, axis = 1)
        
        # extract information from the date
        dat_datetime = []
        for i in tmp_datetime.columns:
            tmp = date_encoding(tmp_datetime[i])
            tmp.columns = i+tmp.columns
            dat_datetime.append(tmp)
            
        dat_datetime = pd.concat(dat_datetime, axis = 1)
        
    # labelencoding y if given
    if y is not None:
        out_y = y
        if y.dtype == 'O':
            #logger.info('Start with label encoding of the target...')
            print('Label encoding the given target', file = file)
            out_y = pd.Series(LabelEncoder().fit_transform(y), index = y.index)
            #logger.info('...End with Target encoding')
    
    # select the target columns
    dat_categ = X.select_dtypes(include=['object'])
    dat_numeric = X.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    
    # placeholder for kterm feature
    dat_kterm = pd.DataFrame(index = X.index)
    
    # generate kterm of categ features
    if k_term:
        print('K-term function is activated, try to extract the k-term for each object columns', file = file)
        
        for i in dat_categ.columns:
            # get k-term feature
            tmp = X[i].value_counts()
            dat_kterm[i + '_kterm'] = X[i].map(lambda x: tmp[x] if x in tmp.index else 0)
        
        print(f'{bcolors.HEADER}{bcolors.BOLD}{dat_kterm.shape[1]}{bcolors.ENDC} k-term features are extracted', file = file)
            
    # onehot encoding and label encoding
    dat_categ_onehot = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values < 8]
    dat_categ_label = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values >= 8]
    flag_onehot = False
    flag_label = False
    
    oe = OneHotEncoder(drop='first')
    
    # oe
    if dat_categ_onehot.shape[1] > 0:
        print('Start to do the one-hot encoding for the following categoric features: '+ (str(dat_categ_onehot.columns.to_list())) + '...', file = file)
        dat_categ_onehot = dat_categ_onehot.fillna('NULL')
        dat_onehot = pd.DataFrame(oe.fit_transform(dat_categ_onehot.astype(str)).toarray(), 
                                  columns=oe.get_feature_names(dat_categ_onehot.columns))
        #logger.info('...End with one-hot encoding')
        flag_onehot = True
        
    else:
        
        dat_onehot = None
        
    # le
    if dat_categ_label.shape[1] > 0:
        print('Start to do the label encoding for the following categoric features: %s...' %(str(dat_categ_label.columns.to_list())), file = file)
        dat_categ_label = dat_categ_label.fillna('NULL')
        dat_label = pd.DataFrame(columns=dat_categ_label.columns)
        for i in dat_categ_label.columns:
            dat_label[i] = LabelEncoder().fit_transform(dat_categ_label[i].astype(str))
        flag_label = True
        #logger.info('...End with label encoding')
        
    else:
        dat_label = None
        
    # combine encoding result
    dat_new = pd.DataFrame()
    if flag_onehot and flag_label:
         dat_new = pd.concat([dat_numeric, dat_onehot, dat_label], axis=1)
    elif flag_onehot:
         dat_new = pd.concat([dat_numeric, dat_onehot], axis=1)
    elif flag_label:
        dat_new = pd.concat([dat_numeric, dat_label], axis=1)
    else:
        dat_new = dat_numeric
    
    # combine k-term result 
    if k_term:
        dat_new = pd.concat([dat_new, dat_kterm], axis = 1)
    
    # combine date time result
    if tmp_datetime.shape[1] >0:
        dat_new = pd.concat([dat_datetime, dat_new], axis = 1)
        
    print(f'Shape of the given dataframe after encoding is: {bcolors.HEADER}{bcolors.BOLD}{dat_new.shape}{bcolors.ENDC}', file = file)
    logger.info('...End with categorical feature transformation')
    
    if y is not None:
        return dat_new, out_y
    else:
        return dat_new
    
def date_encoding(value: pd.Series) -> pd.DataFrame:
    #logger.info(f'Start to extract datetime information from: {value.name}...')
    tmp = pd.to_datetime(value)
    out = tmp.apply(lambda x: pd.Series([x.year, x.month, x.day, x.dayofyear, x.dayofweek,
                                         x.hour, x.minute, x.second, x.microsecond], 
                                        index = [value.name +'_'+ i for i in ['year', 'month', 'day', 'dayofyear', 'dayofweek', 'hour',
                                                                              'minute', 'second','microsecond']]))
    #logger.info('...End with date time information extraction')
    return out

def dt_transform(df: pd.DataFrame):
    from pandas.errors import ParserError
    
    out = []
    for c in df.columns[df.dtypes=='object']: #don't cnvt num
        try:
            out.append(pd.to_datetime(df[c]))
            
        except (ParserError,ValueError): #Can't cnvrt some
            pass # ...so leave whole column as-is unconverted
    
    if len(out) > 0:
        out = pd.concat(out, axis = 1)
    else:
        out = pd.DataFrame()
        
    return out
