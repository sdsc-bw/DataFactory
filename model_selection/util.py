root = '../'
import sys
sys.path.insert(0, root + "codes")

import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_covtype
from sklearn.utils import shuffle
from tsai.all import *
computer_setup()

from DataFactory import DataFactory

def compare_models(models:list):
    
    datafactory = DataFactory()
    
    # load titanic dataset
    df_titanic = pd.read_csv('../data/titanic.csv')
    dfx_titanic, dfy_titanic = datafactory.preprocess(df_titanic, y_col='survived')
    dfx_titanic = shuffle(dfx_titanic)
    scores_titanic = dict() 
    for m in models:
        if m == 'inception_time' or m == 'res_net':
            scores_titanic[m] = datafactory.train_and_evaluate_network(dfx_titanic, dfy_titanic, model=m, mtype="C", epochs=200).recorder.values[-1][2]
        else:       
            _, scores_titanic[m] = datafactory._finetune_sklearn(dfx_titanic, dfy_titanic, model=m,  mtype="C")
    
    # load iris dataset
    data = load_iris()
    df_iris = pd.DataFrame(data.data, columns=data.feature_names)
    df_iris['species'] = pd.Series(data.target)
    df_iris = shuffle(df_iris)
    dfx_iris, dfy_iris = datafactory.preprocess(df_iris, y_col='species')
    scores_iris = dict() 
    for m in models:
        if m == 'inception_time' or m == 'res_net':
            scores_iris[m] = datafactory.train_and_evaluate_network(dfx_iris, dfy_iris, model=m, mtype="C", epochs=100).recorder.values[-1][2]
        else:        
            _, scores_iris[m] = datafactory._finetune_sklearn(dfx_iris, dfy_iris, model=m, mtype="C")
    
    # load wine dataset
    data = load_wine()
    df_wine = pd.DataFrame(data.data, columns=data.feature_names)
    df_wine['class'] = pd.Series(data.target)
    df_wine = shuffle(df_wine)
    dfx_wine, dfy_wine = datafactory.preprocess(df_wine, y_col='class')
    scores_wine = dict() 
    for m in models:
        if m == 'inception_time' or m == 'res_net':
            scores_wine[m] = datafactory.train_and_evaluate_network(dfx_wine, dfy_wine, model=m, mtype="C", epochs=200).recorder.values[-1][2]
        else:
            _, scores_wine[m] = datafactory._finetune_sklearn(dfx_wine, dfy_wine, model=m, mtype="C")
    
    # load covertype dataset
    data = fetch_covtype()
    df_covtype = pd.DataFrame(data.data, columns=data.feature_names)
    df_covtype['type'] = pd.Series(data.target)
    df_covtype = shuffle(df_covtype)
    dfx_covertype, dfy_covertype = datafactory.preprocess(df_covtype, y_col='type')
    scores_covtype = dict() 
    for m in models:
        if m == 'svm' or m == 'knn':
            scores_covtype[m] = 0.0
            continue
        elif m == 'inception_time' or m == 'res_net':
            scores_covtype[m] = 0.0
            continue
            #scores_covtype[m] = datafactory.train_and_evaluate_network(dfx_covertype, dfy_covertype, model=m, mtype="C", epochs=200).recorder.values[-1][2]
        else:  
            _, scores_covtype[m] = datafactory._finetune_sklearn(dfx_covertype, dfy_covertype, model=m, mtype="C")
    
    df = pd.DataFrame(columns=['Models',  'Titanic Dataset',  'Iris Dataset', 'Wine Dataset', 'Covertype Dataset'])
    #df = pd.DataFrame(data,columns=['Models',  'Titanic Dataset',  'Iris Dataset', 'Wine Dataset'])
    for m in models:
        if m == 'decision_tree':
            m_str = 'Decision Tree'
        elif m == 'random_forest':
            m_str = 'Random Forest'
        elif m == 'adaboost':
            m_str = 'AdaBoost'
        elif m == 'gbdt':
            m_str = 'GBDT'
        elif m == 'svm':
            m_str = 'SVM'
        elif m == 'knn':
            m_str = 'KNN'
        elif m == 'res_net':
            m_str = 'ResNet'
        elif m == 'inception_time':
            m_str = 'InceptionTime'
            
        df = df.append({'Models': m_str, 'Titanic Dataset': scores_titanic[m], 'Iris Dataset': scores_iris[m], 'Wine Dataset': scores_wine[m], 'Covertype Dataset': scores_covtype[m]}, ignore_index=True)
        #df = df.append({'Models': m_str, 'Titanic Dataset': scores_titanic[m], 'Iris Dataset': scores_iris[m], 'Wine Dataset': scores_wine[m]}, ignore_index=True)
    return df
    
def compare_networks(models:list):
    
    transforms = [TSStandardize(by_sample=True), TSMagScale(), TSWindowWarp()]
    datafactory = DataFactory()
    
    X_natops, y_natops, splits_natops = get_UCR_data('NATOPS', return_split=False)
    scorse_natops = dict()
    for m in models:
        learn = datafactory.train_and_evaluate_network(X_natops, y_natops, model=m, splits=splits_natops, transforms=transforms)
        scorse_natops[m] = learn.recorder.values[-1][2]
    
    X_oo, y_oo, splits_oo = get_UCR_data('OliveOil', return_split=False)
    scorse_oo = dict()
    for m in models:
        learn = datafactory.train_and_evaluate_network(X_oo, y_oo, model=m, splits=splits_oo, transforms=transforms)
        scorse_oo[m] = learn.recorder.values[-1][2]
    
    X_lsst, y_lsst, splits_lsst = get_UCR_data('LSST', return_split=False)
    scorse_lsst = dict()
    for m in models:
        learn = datafactory.train_and_evaluate_network(X_lsst, y_lsst, model=m, splits=splits_lsst, transforms=transforms)
        scorse_lsst[m] = learn.recorder.values[-1][2]
        
        
    df = pd.DataFrame(columns=['Models',  'NATOPS Dataset',  'OliveOil Dataset', 'LSST Dataset'])
    for m in models:
        if m == 'inception_time':
            m_str = 'InceptionTime'
        elif m == 'inception_time_plus':
            m_str = 'InceptionTimePlus'
        elif m == 'lstm':
            m_str = 'LSTM'
        elif m == 'gru':
            m_str = 'GRU'
        elif m == 'mlp':
            m_str = 'MLP'
        elif m == 'fcn':
            m_str = 'FCN'
        elif m == 'res_net':
            m_str = 'ResNet'
        elif m == 'lstm_fcn':
            m_str = 'LSTM-FCN'
        elif m == 'gru_fcn':
            m_str = 'GRU-FCN'
        elif m == 'mwdn':
            m_str = 'mWDN'
        elif m == 'tcn':
            m_str = 'TCN'
        elif m == 'xception_time':
            m_str = 'XceptionTime'
        elif m == 'res_cnn':
            m_str = 'ResCNN'
        elif m == 'tab_model':
            m_str = 'TabModel'
        elif m == 'omni_scale':
            m_str = 'OmniScale'
        elif m == 'tst':
            m_str = 'TST'
        elif m == 'tab_transformer':
            m_str = 'TabTransformer'
        elif m == 'xcm':
            m_str = 'XCM'
        elif m == 'mini_rocket':
            m_str = 'MiniRocket'     
        
        df = df.append({'Models': m_str, 'NATOPS Dataset': scorse_natops[m], 'OliveOil Dataset': scorse_oo[m], 'LSST Dataset': scorse_lsst[m]}, ignore_index=True)
    return df