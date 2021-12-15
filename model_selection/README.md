# Model Selection
## What is model selection
There is a variety of models that can be used in machine learning like decision trees, random forests, neural networks...
According to the problem, some models fit better than others. If we have labeled data we can use supervised learning methods, whereas if we don't have the labels, we have to use other methods. Here a small overview:

![alt text](https://github.com/sdsc-bw/DataFactory/blob/develop/images/model_selection.png)

Also the models can perform vary on different tasks. For example, for a simple problem it makes sense to use a more simple model like a decision tree, because more complex models like neural networks can lead to overfitting. Whereas these complexe models perform better at non-linear problems.

In this Demo we want to present commonly used machine learning models and how they perform on different datasets.The models supported in this project are and how they can be adressed (C: Classifiction, R: Regression, F: Forecasting):

| Model             | String                | Classification     | Regression         | Forecasting        | Domain  |    Hyperparameters |
| ----------------- | --------------------- | ------------------ | ------------------ | -------------------| ------- | ------------------ |
| Decision Tree     | *decision_tree*       | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C:[see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) R:[see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
| Random Forest     | *random_forest*       | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C:[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) R:[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
| AdaBoost          | *adaboost*            | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C:[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) R:[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
| KNN               | *knn*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C:[see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) R:[see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
| GBDT              | *gbdt*                | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C:[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) R:[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
| GaussianNB        | *gaussian_nb*         | :heavy_check_mark: | :x:                | :x:                | TS/CV   | [see](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
| SVM               | *svm*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C:[see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) R:[see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
| BayesianRidge     | *bayesian*            | :x:                | :heavy_check_mark: | :x:                | TS/CV   | [see](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
| LSTM              | *lstm*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
| GRU               | *gru*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
| MLP               | *mlp*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MLP.py)
| FCN               | *fcn*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py)
| ResNet            | *res_net*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py)
| LSTM-FCN          | *lstm_fcn*            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| GRU-FCN           | *gru_fcn*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| mWDN              | *mwdn*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/mWDN.py)
| TCN               | *tcn*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TCN.py)
| MLSTM-FCN         | *mlstm_fcn*           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| InceptionTime     | *inception_time*      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py)
| InceptionTimePlus | *inception_time_plus* | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTimePlus.py)
| XcetptinTime      | *xception_time*       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XceptionTime.py)
| ResCNN            | *res_cnn*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py)
| TabModel          | *tab_model*           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabModel.py)
| OmniScale         | *omni_scale*          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/OmniScaleCNN.py)
| TST               | *tst*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py)
| MiniRocket        | *mini_rocket*         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKET.py)
| XCM               | *xcm*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see] (https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py)