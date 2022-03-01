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

def check_data_and_distribute(dat: pd.DataFrame, art: str = 'R', y: str = None, file = None, logger = None):
    """
    check the quality of the given data
    if art == 'C', will do labelencoding first for the target column
    ================
    Parameters:
    ================
    dat - type of DataFrame
    art - type of string
        either C for classifcation of R for regression. indicates the type of problem 
    y - type of string
        the name of the target columns; if None, set the last columns of the data set as target
    file - tyoe of file
        the file to output the analyse result
    logger - type of Logger
    =================
    Outputs:
    =================
    dat_numeric - type of DataFrame 
        the dataframe that contains the numeric features
    dat_category - type of pd.Series
        the dataframe that contains the category features
    dat_y - type of DataFrame
        the serie that contains the target
    le_name_mapping - type of dict
        the dict that contain the mapping between the original class in the target to the new class
    flag_balance - type of bool
        if art "C", return if the given dataset is balanced or not. if not "C", return false
    flag_wrong_target - type of bool
        it is True, when there are more then 10 classes contain less than 10 items
    """
    print('#'*30, file = file)
    print('Check data', file = file)
    print('#'*30, file = file)
    if logger:
        logger.info('Start to check the given data set...')
        
    dat = dat.reset_index().iloc[:, 1:]

    ## split features and target
    if y:
        # check if y exist
        while y not in dat.columns():
            y = print('Given target not found, please input the new target name:')
        
        # check na in y and drop na
        number_na_in_target = dat[y].isna().sum()
        if number_na_in_target > 0:
            dat = dat[dat[y].isna().map(lambda x: not x)]

        dat_y = dat[y]
        cols = dat.columns.to_list()
        cols.remove(y)
        dat_x = dat[cols]
    else:
        print('No target name given, use the last column in the given dataset as target.')

        # check na in y and drop na
        number_na_in_target = dat.iloc[:, -1].isna().sum()
        if number_na_in_target > 0:
            dat = dat[dat.iloc[:, -1].isna().map(lambda x: not x)]

        dat_y = dat[dat.columns[-1]]
        dat_x = dat[dat.columns[:-1]]
    
    ## basic report including following information in form of logger info (? print?):
    # - art of task C/R
    # - number of features, list ten of the features
    # - list of nummeric feature and 
    # - list of character feature 
    # - the type of target feature and number of different classes
    dat_category = dat_x.select_dtypes(include = ['object'])
    dat_numeric =  dat_x.select_dtypes(include=['float32', 'float64', 'int'])

    print(f'#### basic information ####', file = file)
    print(f'The type of the task is:{bcolors.HEADER}{bcolors.BOLD}{"Classification" if art == "C" else "Regression"}{bcolors.ENDC}, with target feature: {bcolors.HEADER}{bcolors.BOLD}{y if y else dat.columns[-1]}{bcolors.ENDC}.', file = file)
    print(f'The given data include {bcolors.HEADER}{bcolors.BOLD}{dat.shape[1]}{bcolors.ENDC} columns and {bcolors.HEADER}{bcolors.BOLD}{dat.shape[0]}{bcolors.ENDC} rows:', file = file)
    print(f'- {bcolors.HEADER}{bcolors.BOLD}{len(dat_numeric.columns)}{bcolors.ENDC} features are numeric: {dat_numeric.columns[:5].to_list()} {"..." if len(dat_numeric.columns)>5 else "."}', file = file)
    print(f'- {bcolors.HEADER}{bcolors.BOLD}{len(dat_category.columns)}{bcolors.ENDC} features are category: {dat_category.columns[:5].to_list()} {"..." if len(dat_category.columns)>5 else "."}', file = file)

    flag_balance = False
    flag_wrong_target = False

    if art == 'C':
        print(f'- target value has {bcolors.HEADER}{bcolors.BOLD}{len(dat_y.unique())}{bcolors.ENDC} different classed and is type of {bcolors.HEADER}{bcolors.BOLD}{"category" if dat_y.dtype == pd.CategoricalDtype else "numeric"}{bcolors.ENDC}', file = file)

        # collect the class information
        tmp = dat_y.value_counts()

        cls = {}
        number_items_less_10 = 0
        for i in range(tmp.shape[0]):
            cls[tmp.index[i]] = tmp.iloc[i]
        if tmp.iloc[i] < 10:
            number_items_less_10 += 1
      
        # check if the given y is the right target
        if number_items_less_10 >= 10:
            flag_wrong_target = True

        # check balancy
        if tmp.std()<100:
            flag_balance = True
      
        if flag_balance:
            print(f'\t The given dataset is balance: {bcolors.HEADER}{bcolors.BOLD}{cls}{bcolors.ENDC}', file = file)
        else:
            print(f'\t The given dataset is unblance: {bcolors.HEADER}{bcolors.BOLD}{cls}{bcolors.ENDC}', file = file)

    ## report basic flaws of the given data, which include following info:
    # - na and inf in the feature and target columns
    # - feature that contain only one value
    print(f'#### na/inf value ####', file = file)
    # na, inf in y
    dat_y = dat_y.replace(np.inf, np.nan)
    print(f'There is in total {bcolors.HEADER}{bcolors.BOLD}{number_na_in_target}{bcolors.ENDC} NA and Inf value found in the target column, corresponding items are deleted from the given dataset.', file = file)

    # na, inf in x
    dat_x = dat_x.replace(np.inf, np.nan)
    print(f'There is in total {bcolors.HEADER}{bcolors.BOLD}{dat_x.isna().sum().sum()}{bcolors.ENDC} NA and Inf value found in the given features data.', file = file)
    if dat_x.isna().sum().sum() > 0:
        print(dat_x.isna().sum().map(lambda x: str(x) + '/' + str(dat_x.shape[0])), file = file)
    
    # unique: numeric and cataegory
    tmp = dat_numeric.std() == 1
    unique_column_nemeric = tmp.index[tmp].to_list()
    unique_column_category = []

    for col in dat_category.columns:
        tmp_col = dat_category[col]
        if len(tmp_col.unique()) == 1:
            unique_column_category.append(col)

    ## deal with the categoric target
    # 
    le_name_mapping = None
    if art == 'C':
        if logger:
            logger.info('- start to label target feature y for classification task')
        print(f'#### Labeling the target ####', file = file)
        print('Relabel the target class to make it as type of numeric and start from 0', file = file)
        le = LabelEncoder()
        dat_y = le.fit_transform(dat_y)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f'- {le_name_mapping}', file = file)
        # check balanced
        #dat = self._balanced_sampling(dat)
        if logger:
            logger.info('+ end with label encoding the target feature')
    
    if logger:
        logger.info('...finish with data check')

    return dat_numeric, dat_category, dat_y, le_name_mapping, flag_balance, flag_wrong_target
