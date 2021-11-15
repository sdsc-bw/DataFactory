root = '../'
import sys
sys.path.insert(0, root + "codes")

import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_covtype

from DataFactory import DataFactory

def compare_models(models:list):
    
    datafactory = DataFactory()
    
    # load titanic dataset
    df_titanic = pd.read_csv('../data/titanic.csv')
    dfx_titanic, dfy_titanic = datafactory.preprocess(df_titanic, y_col='survived')
    scores_titanic = dict() 
    for m in models:
        _, scores_titanic[m] = datafactory.train_and_evaluate(dfx_titanic, dfy_titanic, strategy='random', model=m,  mtype="C")
    
    # load iris dataset
    data = load_iris()
    df_iris = pd.DataFrame(data.data, columns=data.feature_names)
    df_iris['species'] = pd.Series(data.target)
    dfx_iris, dfy_iris = datafactory.preprocess(df_iris, y_col='species')
    scores_iris = dict() 
    for m in models:
        _, scores_iris[m] = datafactory.train_and_evaluate(dfx_iris, dfy_iris, strategy='random', model=m, mtype="C")
    
    # load wine dataset
    data = load_wine()
    df_wine = pd.DataFrame(data.data, columns=data.feature_names)
    df_wine['class'] = pd.Series(data.target)
    dfx_wine, dfy_wine = datafactory.preprocess(df_wine, y_col='class')
    scores_wine = dict() 
    for m in models:
        _, scores_wine[m] = datafactory.train_and_evaluate(dfx_wine, dfy_wine, strategy='random', model=m, mtype="C")
    
    # load covertype dataset
    #data = fetch_covtype()
    #df_covtype = pd.DataFrame(data.data, columns=data.feature_names)
    #df_covtype['type'] = pd.Series(data.target)
    #dfx_covertype, dfy_covertype = datafactory.preprocess(df_covtype, y_col='type')
    #scores_covtype = dict() 
    #for m in models:
    #    _, scores_covtype[m] = datafactory.train_and_evaluate(dfx_covertype, dfy_covertype, model=m, mtype="C")
    
    #df = pd.DataFrame(data,columns=['Models',  'Titanic Dataset',  'Iris Dataset', 'Wine Dataset', 'Covertype Dataset'])
    df = pd.DataFrame(data,columns=['Models',  'Titanic Dataset',  'Iris Dataset', 'Wine Dataset'])
    for m in models:
        if m == 'decision_tree':
            m_str = 'Decision Tree'
        elif m == 'random_forest':
            m_str = 'Random Forest'
        elif m == 'adaboost':
            m_str = 'AdaBoost'
        elif m == 'gbdt':
            m_str = 'GBDT'
            
        #df = df.append({'Models': m_str, 'Titanic Dataset': scores_titanic[m], 'Iris Dataset': scores_iris[m], 'Wine Dataset': scores_wine[m], 'Covertype Dataset': scores_covtype[m]}, ignore_index=True)
        df = df.append({'Models': m_str, 'Titanic Dataset': scores_titanic[m], 'Iris Dataset': scores_iris[m], 'Wine Dataset': scores_wine[m]}, ignore_index=True)
    return df
    