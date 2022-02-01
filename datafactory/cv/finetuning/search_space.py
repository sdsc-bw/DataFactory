'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from hyperopt import hp

######## sklearn
std_decision_tree = {'model': 'decision_tree', 
                      'max_depth': hp.quniform('max_depth_dt', 1, 10, 1), 
                      'criterion': hp.choice('criterion_dt', ['gini', 'entropy']), 
                      'min_samples_split': hp.choice('min_samples_split_dt',[2, 3, 5]), 
                      'min_samples_leaf': hp.choice('min_samples_leaf_dt', [1, 2, 4])}
std_random_forest = {'model': 'random_forest', 
                     'max_depth': hp.choice('max_depth_rf', [1, 2, 3, 5, 10, 20, 50]), 
                     'min_samples_leaf': hp.choice('min_samples_leaf_rf', [1, 2, 4]), 
                     'min_samples_split': hp.choice('min_samples_split_rf', [2, 5, 10]), 
                     'n_estimators': hp.choice('n_estimators_rf', [50, 100, 200])}
std_ada_boost = {'model': 'ada_boost', 
                 'n_estimators': hp.choice('n_estimators_ab', [50, 100, 200]), 
                 'learning_rate': hp.choice('learning_rate_ab', [0.001, 0.01, 0.1])}
std_knn = {'model': 'knn', 
           'n_neighbors': hp.quniform('n_neighbors_knn', 1, 30, 1),
           'weights': hp.choice('weights_knn', ["uniform", "distance"])}
std_gbdt = {'model': 'gbdt', 
            'max_depth': hp.choice('max_depth_gbdt', [1, 2, 3, 5, 10, 20, 50]), 
            'learning_rate': hp.choice('learning_rate_gbdt', [0.001, 0.01, 0.1]), 
            'min_samples_leaf': hp.quniform('min_samples_leaf_gbdt', 1, 5, 1)}
std_gaussian_nb =  {'model': 'gaussian_nb', 
                    'var_smoothing': hp.lognormal('var_smoothing_nb', 0, 1.0)}
std_svm = {'model': 'svm', 
           'C': hp.lognormal('c_svm', 0, 1.0), 
           'kernel': hp.choice('kernel_svm', ['linear', 'rbf'])}
std_bayesian =  {'model': 'bayesian', 
                 'alpha_1': hp.choice('alpha_1_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]), 
                 'alpha_2': hp.choice('alpha_2_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]), 
                 'lambda_1': hp.choice('lambda_1_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]), 
                 'lambda_2': hp.choice('lambda_2_bay', [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])}

######### pytorchcv
std_res_net = {'model': 'res_net',
               'epochs': 25,
               'lr_max': 1e-3,
               'layers': hp.choice('layers_res_net', [10, 14, 18])}

######## Standard Searchspace
std_search_space = {'decision_tree': std_decision_tree,
                    'random_forest': std_random_forest, 
                    'ada_boost': std_ada_boost, 
                    'knn': std_knn, 
                    'gbdt': std_gbdt,
                    'gaussian_nb': std_gaussian_nb, 
                    'svm': std_svm, 
                    'bayesian': std_bayesian,
                    'res_net': std_res_net}
