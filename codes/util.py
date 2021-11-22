import sklearn

def model_to_string(model):
    if type(model) == sklearn.tree._classes.DecisionTreeClassifier or type(model) == sklearn.tree._classes.DecisionTreeRegressor:
        return 'decision_tree'
    elif  type(model) == sklearn.ensemble._forest.RandomForestClassifier or type(model) == sklearn.ensemble._forest.RandomForestRegressor:
        return 'random_forest'
    elif type(model) == sklearn.ensemble._weight_boosting.AdaBoostClassifier or type(model) == sklearn.ensemble._weight_boosting.AdaBoostRegressor:
        return 'adaboost'
    elif type(model) == sklearn.neighbors._classification.KNeighborsClassifier or type(model) == sklearn.neighbors._classification.KNeighborsRegressor:
        return 'knn'
    elif type(model) == sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier or type(model) == sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor:
        return 'gbdt'
    elif type(model) == sklearn.naive_bayes.GaussianNB:
        return 'gaussian_nb'
    elif  type(model) == sklearn.svm._classes.SVC == sklearn.svm._classes.SVR:
        return 'svm'
    elif type(model) == sklearn.linear_model._bayes.BayesianRidge:
        return 'bayesian'