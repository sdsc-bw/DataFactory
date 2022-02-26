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
               'epochs': 10,
               'lr_max': 1e-3,
               'n_layers': hp.choice('n_layers_res_net', [10, 14, 16]), 
               'down_sampling': hp.choice('down_sampling_res_net', [True, False])}
std_se_res_net = {'model': 'se_res_net',
                  'epochs': 10,
                  'lr_max': 1e-3,
                  'n_layers': hp.choice('n_layers_se_res_net', [10, 14, 16])}
std_res_next = {'model': 'res_next',
                'epochs': 10,
                'lr_max': 1e-3,
                'n_blocks': hp.choice('n_blocks_res_next', [14, 26]), 
                'cardinality': hp.choice('cardinality_res_next', [16, 32]), 
                'bottleneck_width': hp.choice('bottleneck_width_res_next', [2, 4])}
std_alex_net = {'model': 'alex_net',
               'epochs': 10,
               'lr_max': 1e-3,
               'version': hp.choice('version_alex_net', ['a', 'b'])}
std_vgg = {'model': 'vgg',
           'epochs': 10,
           'lr_max': 1e-3,
           'n_layers': hp.choice('n_layers_vgg', [11, 13, 16]), 
           'batch_norm': hp.choice('bn_vgg', [True, False])}
std_efficient_net = {'model': 'efficient_net',
                     'epochs': 10,
                     'lr_max': 1e-3}
std_wrn = {'model': 'wrn',
           'epochs': 10,
           'lr_max': 1e-3}
std_reg_net = {'model': 'reg_net',
               'epochs': 10,
               'lr_max': 1e-3,
               'version': hp.choice('version_reg_net', ['x', 'y']), 
               'n_mf': hp.choice('n_mf', [200, 400])}
std_sc_net = {'model': 'sc_net',
              'epochs': 10,
              'lr_max': 1e-3,
              'n_layers': hp.choice('n_layers_sc_net', [50, 101]), 
              'down_sampling': hp.choice('down_sampling_sc_net', [True, False])} 
std_pnas_net = {'model': 'pnas_net',
                'epochs': 10,
                'lr_max': 1e-3} 

######## Standard Searchspace
std_search_space = {'decision_tree': std_decision_tree,
                    'random_forest': std_random_forest, 
                    'ada_boost': std_ada_boost, 
                    'knn': std_knn, 
                    'gbdt': std_gbdt,
                    'gaussian_nb': std_gaussian_nb, 
                    'svm': std_svm, 
                    'bayesian': std_bayesian,
                    'res_net': std_res_net, 
                    'se_res_net': std_se_res_net,
                    'res_next': std_res_next,
                    'alex_net': std_alex_net, 
                    'vgg': std_vgg, 
                    'efficient_net': std_efficient_net, 
                    'wrn': std_wrn, 
                    'reg_net': std_reg_net, 
                    'sc_net': std_sc_net,
                    'pnas_net': std_pnas_net}
