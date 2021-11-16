import numpy as np

std_dt = {"criterion": ['gini', 'entropy'], "max_depth": range(1, 10), "min_samples_split": range(1, 10), "min_samples_leaf": range(1, 5)}
std_rf = {'max_depth': [1, 2, 3, 5, 10, 20, 50], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10], 'n_estimators': [50, 100, 200]}
std_adaboost = {'n_estimators': [50, 100, 200], 'learning_rate':[0.001,0.01,.1]}
std_knn = {'n_neighbors': range(1, 30), 'weights':["uniform", "distance"]}
std_gbdt = {'max_depth': [1, 2, 3, 5, 10, 20, 50], 'learning_rate':[0.001,0.01,.1], 'max_depth': [1, 2, 3, 5, 10, 20, 50, None], 'min_samples_leaf': [1, 2, 4]}
std_gaussian_nb = {'var_smoothing': np.logspace(0,-9, num=100)}
std_svm ={'C': [0.0, 10], 'gamma': [1, 0.1, 0.001],'kernel': ['linear', 'rbf']}
std_bayesian = {'alpha_1': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'alpha_2': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'lambda_1': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], 'lambda_2': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]}

def get_std_search_space(model):
    if model == 'decision_tree':
        return std_dt
    elif model == 'random_forest':
        return std_rf 
    elif model == 'adaboost':
        return std_adaboost
    elif model == 'knn':
        return std_knn
    elif model == 'gbdt':
        return std_gbdt
    elif model == 'gaussian_nb':
        return std_gaussian_nb
    elif model == 'svm':
        return std_svm
    elif model == 'bayesian':
        return std_bayesian
    else: 
        return std_dt