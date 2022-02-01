'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from hyperopt import hp

######## sklearn
std_decision_tree_c = {'model': 'decision_tree', 
                      'max_depth': hp.quniform('max_depth_dt', 1, 10, 1), 
                      'criterion': hp.choice('criterion_dt', ['gini', 'entropy']), 
                      'min_samples_split': hp.choice('min_samples_split_dt',[2, 3, 5]), 
                      'min_samples_leaf': hp.choice('min_samples_leaf_dt', [1, 2, 4])}
std_decision_tree_r = {'model': 'decision_tree', 
                       'max_depth': hp.quniform('max_depth_dt', 1, 10, 1), 
                       'criterion': hp.choice('criterion_dt', ['squared_error', 'absolute_error']), 
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

######### tsai
std_inception_time = {'model': 'inception_time',
                      'nb_filters': hp.choice('nb_filters_it', [32, 64, 96, 128]), 
                      'nf': hp.choice('nf_it', [32, 64]), 
                      'epochs': 25,
                      'lr_max': 1e-3, 
                      'metrics': ['accuracy']}

std_inception_time_plus = {'model': 'inception_time_plus',
                           'nb_filters': hp.choice('nb_filters_itp', [32, 64, 96, 128]), 
                           'fc_dropout': hp.choice('fc_dropout_ipt', [0.0, 0.3]), 
                           'epochs': 25,
                           'lr_max': 1e-3,
                           'metrics': ['accuracy']}
std_lstm = {'model': 'lstm',
            'n_layers': hp.choice('n_layers_lstm', [1, 2, 3]), 
            'bidirectional': hp.choice('bidirectional_lstm', [True, False]), 
            'epochs': 25,
            'lr_max': 1e-3,
            'metrics': ['accuracy']}
std_gru = {'model': 'gru',
           'epochs': 25,
           'lr_max': 1e-3,
           'metrics': ['accuracy']}
std_mlp = {'model': 'mlp',
           'use_bn': hp.choice('use_bn_mlp', [True, False]), 
           'bn_final': hp.choice('bn_final_mlp', [True, False]), 
           'lin_first': hp.choice('lin_first_mlp', [True, False]), 
           'fc_dropout': hp.choice('fc_dropout_mlp', [0.0, 0.3]), 
           'epochs': 25,
           'lr_max': 1e-3,
           'metrics': ['accuracy']}
std_fcn = {'model': 'fcn',
           'epochs': 25,
           'lr_max': 1e-3,
           'metrics': ['accuracy']}
std_res_net = {'model': 'res_net',
               'epochs': 25,
               'lr_max': 1e-3,
               'metrics': ['accuracy']}
std_lstm_fcn = {'model': 'lstm_fcn',
                'cell_dropout': hp.choice('cell_dropout_lstm_fcn', [0.0, 0.3, 0.5]),
                'rnn_dropout': hp.choice('rnn_dropout_lstm_fcn', [0.6, 0.8, 0.9]),
                'fc_dropout': hp.choice('fc_dropout_lstm_fcn', [0.0, 0.3, 0.5]),
                'bidirectional': hp.choice('bidirectional_lstm_fcn', [True, False]),
                'shuffle': hp.choice('shuffle_lstm_fcn', [True, False]),
                'epochs': 25,
                'lr_max': 1e-3,
                'metrics': ['accuracy']}
std_gru_fcn = {'model': 'gru_fcn',
               'cell_dropout': hp.choice('cell_dropout_gru_fcn', [0.0, 0.3, 0.5]),
               'rnn_dropout': hp.choice('rnn_dropout_gru_fcn', [0.6, 0.8, 0.9]),
               'fc_dropout': hp.choice('fc_dropout_gru_fcn', [0.0, 0.3, 0.5]),
               'bidirectional': hp.choice('bidirectional_gru_fcn', [True, False]),
               'shuffle': hp.choice('shuffle_gru_fcn', [True, False]),
               'epochs': 25,
               'lr_max': 1e-3,
               'metrics': ['accuracy']}
std_mwdn = {'model': 'mwdn',
            'epochs': 25,
            'lr_max': 1e-3,
            'metrics': ['accuracy']}
std_tcn = {'model': 'tcn',
           'epochs': 25,
           'lr_max': 1e-3,
           'metrics': ['accuracy']}
std_xception_time = {'model': 'xception_time',
                     'epochs': 25,
                     'lr_max': 1e-3,
                     'metrics': ['accuracy']}
std_res_cnn = {'model': 'res_cnn',
               'epochs': 25,
               'lr_max': 1e-3,
               'metrics': ['accuracy']}
std_tab_model = {'model': 'tab_model',
                 'epochs': 25,
                 'lr_max': 1e-3,
                 'metrics': ['accuracy']}
std_omni_scale = {'model': 'omni_scale',
                  'paramenter_number_of_layer_list': hp.choice('paramenter_number_of_layer_list_omni_scale', [[8 * 128, 5 * 128 * 256 + 2 * 256 * 128]]), 
                  'few_shot': hp.choice('few_shot_omni_scale', [True, False]), 
                  'epochs': 25,
                  'lr_max': 1e-3,
                  'metrics': ['accuracy']}
std_tst= {'model': 'tst',
          'n_layers': hp.choice('n_layers_tst', [2, 3, 4]), 
          'd_model': hp.choice('d_model_tst', [64, 128, 256]), 
          'n_heady': hp.choice('n_heads_tst', [8, 16, 32]), 
          'd_ff': hp.choice('d_ff_tst', [64, 128, 256]), 
          'dropout': hp.choice('dropout_tst', [0.1, 0.3]), 
          'act': hp.choice('act_tst', ['relu', 'gelu']), 
          'fc_dropout': hp.choice('fc_dropout_tst', [0.3, 0.5]), 
          'epochs': 25,
          'lr_max': 1e-3,
          'metrics': ['accuracy']}
std_mini_rocket= {'model': 'mini_rocket',
                  'num_features': hp.choice('num_features_mini_rocket', [5000, 10000, 15000]), 
                  'max_dilations_per_kernel': hp.choice('max_dilations_per_kernel_mini_rocket', [16, 32, 64]), 
                  'bn': hp.choice('bn_mini_rocket', [True, False]), 
                  'fc_dropout': hp.choice('fc_dropout_mini_rocket', [0.3, 0.5]), 
                  'epochs': 25,
                  'lr_max': 1e-3,
                  'metrics': ['accuracy']}

######## Standard Searchspace
std_search_space = {'decision_tree_c': std_decision_tree_c, 
                    'decision_tree_r': std_decision_tree_r,
                    'random_forest': std_random_forest, 
                    'ada_boost': std_ada_boost, 
                    'knn': std_knn, 
                    'gbdt': std_gbdt,
                    'gaussian_nb': std_gaussian_nb, 
                    'svm': std_svm, 
                    'bayesian': std_bayesian, 
                    'inception_time': std_inception_time, 
                    'inception_time_plus': std_inception_time_plus, 
                    'lstm': std_lstm, 
                    'gru': std_gru, 
                    'mlp': std_mlp, 
                    'fcn': std_fcn, 
                    'res_net': std_res_net, 
                    'lstm_fcn': std_lstm_fcn, 
                    'gru_fcn': std_gru_fcn, 
                    'mwdn': std_mwdn, 
                    'tcn': std_tcn, 
                    'xception_time': std_xception_time, 
                    'res_cnn': std_res_cnn, 
                    'tab_model': std_tab_model, 
                    'omni_scale': std_omni_scale, 
                    'tst': std_tst, 
                    'mini_rocket': std_mini_rocket}
