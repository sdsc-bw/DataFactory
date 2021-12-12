import numpy as np
from hyperopt import tpe, hp
import sys
from tsai.all import *
computer_setup()

dt = 'decision_tree'
rf = 'random_forest'
ab = 'adaboost'
knn = 'knn'
gbdt = 'gbdt'
nb = 'gaussian_nb'
svm = 'svm'
bay = 'bayesian'
it = 'inception_time'
itp = 'inception_time_plus'

learner = 'learner'

# sklearn standard search space
sk_std_dt = {"criterion": ['gini', 'entropy'], "max_depth": range(1, 10), "min_samples_split": range(1, 10), "min_samples_leaf": range(1, 5)}
sk_std_rf = {'max_depth': [1, 2, 3, 5, 10, 20, 50], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10], 'n_estimators': [50, 100, 200]}
sk_std_adaboost = {'n_estimators': [50, 100, 200], 'learning_rate':[0.001,0.01,.1]}
sk_std_knn = {'n_neighbors': range(1, 30), 'weights':["uniform", "distance"]}
sk_std_gbdt = {'max_depth': [1, 2, 3, 5, 10, 20, 50], 'learning_rate':[0.001,0.01,.1], 'max_depth': [1, 2, 3, 5, 10, 20, 50, None], 'min_samples_leaf': [1, 2, 4]}
sk_std_gaussian_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
sk_std_svm ={'C': [0.0, 10], 'gamma': [1, 0.1, 0.001],'kernel': ['linear', 'rbf']}
sk_std_bayesian = {'alpha_1': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'alpha_2': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'lambda_1': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'lambda_2': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]}

# hyperopt standard search space
hp_std_dt = {'model': dt, 'max_depth': hp.quniform('max_depth_dt', 1, 10, 1), 'criterion': hp.choice('criterion_dt', ['gini', 'entropy']), 'min_samples_split': hp.choice('min_samples_split_dt',[2, 3, 5]), 'min_samples_leaf': hp.choice('min_samples_leaf_dt', [1, 2, 4])}
hp_std_rf = {'model': rf, 'max_depth': hp.choice('max_depth_rf', [1, 2, 3, 5, 10, 20, 50]), 'min_samples_leaf': hp.choice('min_samples_leaf_rf', [1, 2, 4]), 'min_samples_split': hp.choice('min_samples_split_rf', [2, 5, 10]), 'n_estimators': hp.choice('n_estimators_rf', [50, 100, 200])}
hp_std_adaboost = {'model': ab, 'n_estimators': hp.choice('n_estimators_ab', [50, 100, 200]), 'learning_rate': hp.choice('learning_rate_ab', [0.001,0.01,.1])}
hp_std_knn = {'model': knn, 'n_neighbors': hp.quniform('n_neighbors_knn', 1, 30, 1), 'weights': hp.choice('weights_knn', ["uniform", "distance"])}
hp_std_gbdt = {'model': gbdt, 'max_depth': hp.choice('max_depth_gbdt', [1, 2, 3, 5, 10, 20, 50]), 'learning_rate': hp.choice('learning_rate_gbdt', [0.001,0.01,.1]), 'min_samples_leaf': hp.quniform('min_samples_leaf_gbdt', 1, 5, 1)}
hp_std_gaussian_nb =  {'model': nb, 'var_smoothing': hp.lognormal('var_smoothing_nb', 0, 1.0)}
hp_std_svm = {'model': svm, 'C': hp.lognormal('c_svm', 0, 1.0), 'kernel': hp.choice('kernel_svm', ['linear', 'rbf'])}
hp_std_bayesian =  {'model': bay, 'alpha_1': hp.choice('alpha_1_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]), 'alpha_2': hp.choice('alpha_2_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]), 'lambda_1': hp.choice('lambda_1_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]), 'lambda_2': hp.choice('lambda_2_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])}
hp_std_learner = {'loss_func': 'cross_entropy', 'opt_func': 'adam', 'epochs': 25, 'lr_max': 1e-3, 'metrics': ['accuracy']}
hp_std_inception_time = {'nb_filters': hp.choice('nb_filters_itp', [32, 64, 96, 128]), 'nf': hp.choice('nf_it', [32, 64])}
#hp_std_learner = {'epochs': 25, 'lr_max': 1e-3}
#hp_std_inception_time = {}
hp_std_inception_time_plus = {'nb_filters': hp.choice('nb_filters_itp', [32, 64, 96, 128]), 'fc_dropout': hp.choice('fc_dropout_ipt', [0.3, 0.5])}

def get_sklearn_search_space(model: str, params):
    if model in params:
        search_space = params[model]
        del params[model]
        return search_space
    
    if model == dt:
        search_space = sk_std_dt.copy()
    elif model == rf:
        search_space = sk_std_rf.copy()
    elif model == ab:
        search_space = sk_std_adaboost.copy()
    elif model == knn:
        search_space = hp_std_knn.copy()
    elif model == gbdt:
        search_space = sk_std_gbdt.copy()    
    elif model == nb:
        search_space = sk_std_gaussian_nb.copy()
    elif model == svm:
        search_space = sk_std_svm.copy()
    elif model == bay:
        search_space = sk_std_bayesian.copy()
    else: 
        search_space = sk_std_dt.copy()
        
    return search_space

def get_hyperopt_search_space(models: list, mtype:str, cv:int, params):
    search_space_list = []
    if dt in models:
        if dt not in params:
            model_space = hp_std_dt.copy()
        else:
            model_space = params[dt]
            model_space['model'] = dt
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'random_forest' in models:
        if rf not in params:
            model_space = hp_std_rf.copy()
        else:
            model_space = params[rf]
            model_space['model'] = rf
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'adaboost' in models:
        if ab not in params:
            model_space = hp_std_adaboost.copy()
        else:
            model_space = params[ab]
            model_space['model'] = ab
        model_space = hp_std_adaboost.copy()
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'knn' in models:
        if knn not in params:
            model_space = hp_std_knn.copy()
        else:
            model_space = params[knn]
            model_space['model'] = knn
        model_space = hp_std_knn.copy()
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'gbdt' in models:
        if gbdt not in params:
            model_space = hp_std_gbdt.copy()
        else:
            model_space = params[gbdt]
            model_space['model'] = gbdt
        model_space = hp_std_gbdt.copy()
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'gaussian_nb' in models:
        if nb not in params:
            model_space = hp_std_gaussian_nb.copy()
        else:
            model_space = params[nb]
            model_space['model'] = nb
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'svm' in models:
        if svm not in params:
            model_space = hp_std_svm.copy()
        else:
            model_space = params[svm]
            model_space['model'] = svm
        model_space = hp_std_svm.copy()
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'bayesian' in models:   
        if bay not in params:
            model_space = hp_std_bayesian.copy()
        else:
            model_space = params[bay]
            model_space['model'] = bay
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'learner' in models:
        if learner in params:
            model_space = hp_std_learner.copy()
        else:
            model_space = params[learner]
            model_space['model'] = learner
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
    if 'inception_time' in models:
        if it not in params:
            model_space = hp_std_learner.copy()
            model_space['arch_config'] = hp_std_inception_time.copy()
            model_space['model'] = it
        else:
            model_space = params.get('learner', hp_std_learner.copy())
            if 'learner' in params[it]:
                del params[it]['learner']
            model_space['arch_config'] = params[it]
            model_space['model'] = it
        model_space['type'] = mtype
        model_space['cv'] = cv
        search_space_list.append(model_space)
        
    search_space_model = hp.choice('classifier_type', search_space_list)
    return search_space_model