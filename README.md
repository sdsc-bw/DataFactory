# DataFactory
This is a package developed by SDSC engineers for datamining and machine learning. It unites and unifies methods from different packages like [imblearn](https://imbalanced-learn.org/stable/), [sklearn](https://scikit-learn.org/stable/index.html), [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [hyperopt](https://github.com/hyperopt/hyperopt) and [tsai](https://github.com/timeseriesAI/tsai).

## Preprocessing
We offer methods for data preprocessing. This includes label encoding, data balancing, sampling and dealing with NA values and outliers.

## Feature Engineering
In addition to that, we provide functions for feature engineering. This includes unsupervised (unary, binary and multiple) and supervised transformations.

## Finetuning
We also provide methods from some common finetuning packages like [sklearn](https://scikit-learn.org/stable/index.html), [auto-sklearn](https://automl.github.io/auto-sklearn/master/), [hyperopt](https://github.com/hyperopt/hyperopt). (Note that sklearn and auto-sklearn only support sklearn models).

Here is a complete list of our supported models:

| Model             | String                | Classification     | Regression         | Forecasting        | Domain  | Hyperparameters    |
| ----------------- | --------------------- | ------------------ | ------------------ | -------------------| ------- | ------------------ |
| Decision Tree     | *decision_tree*       | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
| Random Forest     | *random_forest*       | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
| AdaBoost          | *ada_boost*            | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
| KNN               | *knn*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
| GBDT              | *gbdt*                | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
| Gaussian NB        | *gaussian_nb*         | :heavy_check_mark: | :x:                | :x:                | TS/CV   | [see](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
| SVM               | *svm*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                | TS/CV   | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
| Bayesian Ridge     | *bayesian*            | :x:                | :heavy_check_mark: | :x:                | TS/CV   | [see](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
| LSTM              | *lstm*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
| GRU               | *gru*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
| MLP               | *mlp*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MLP.py)
| FCN               | *fcn*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py)
| ResNet            | *res_net*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py)
| LSTM-FCN          | *lstm_fcn*            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| GRU-FCN           | *gru_fcn*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| mWDN              | *mwdn*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/mWDN.py)
| TCN               | *tcn*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TCN.py)
| MLSTM-FCN         | *mlstm_fcn*           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| InceptionTime     | *inception_time*      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py)
| InceptionTimePlus | *inception_time_plus* | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTimePlus.py)
| XcetptionTime      | *xception_time*       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XceptionTime.py)
| ResCNN            | *res_cnn*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py)
| TabModel          | *tab_model*           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabModel.py)
| OmniScale         | *omni_scale*          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/OmniScaleCNN.py)
| TST               | *tst*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py)
| MiniRocket        | *mini_rocket*         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKET.py)
| XCM               | *xcm*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | TS      | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py)

(C: Classifiction, R: Regression, F: Forecasting, TS: Time Series, CV: Computer Vision)