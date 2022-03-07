# DataFactory
This Github is mainly used to introduce some commonly used methods, as well as some feature engineering methods independently researched and developed by SDSC staff. It unites and unifies methods from different packages like [imblearn](https://imbalanced-learn.org/stable/), [sklearn](https://scikit-learn.org/stable/index.html), [hyperopt](https://github.com/hyperopt/hyperopt) and [tsai](https://github.com/timeseriesAI/tsai). 
The common dataminig process and how to use the DataFactory for this is shown in our [demos](https://github.com/sdsc-bw/DataFactory/tree/develop/demos).

## Run (Temporary, to be removed)
Go to the root directory and use the following code to create the test report:
``python usersry_01_01_dash.py --datapath=./data/dataset_31_credit-g.csv --outputpath=./results/``

## Preprocessing
We offer methods for data preprocessing. This includes label encoding, data balancing, sampling and dealing with NA values and outliers.

## Feature Engineering
In addition to that, we provide functions for feature engineering. This includes unary, binary and multiple transformations.

## Finetuning
We also provide a finetuning method based on [hyperopt](https://github.com/hyperopt/hyperopt).

Here is a complete list of our supported models for time series:

| Model             | String                | Classification     | Regression         | Forecasting         | Hyperparameters    |
| ----------------- | --------------------- | ------------------ | ------------------ | ------------------- | ------------------ |
| Decision Tree     | *decision_tree*       | :heavy_check_mark: | :heavy_check_mark: | :x:                 | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
| Random Forest     | *random_forest*       | :heavy_check_mark: | :heavy_check_mark: | :x:                 | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
| AdaBoost          | *ada_boost*            | :heavy_check_mark: | :heavy_check_mark: | :x:                | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
| KNN               | *knn*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                 | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
| GBDT              | *gbdt*                | :heavy_check_mark: | :heavy_check_mark: | :x:                 | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
| Gaussian NB        | *gaussian_nb*         | :heavy_check_mark: | :x:                | :x:                | [see](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
| SVM               | *svm*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                 | C: [see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) R: [see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
| Bayesian Ridge     | *bayesian*            | :x:                | :heavy_check_mark: | :x:                | [see](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
| LSTM              | *lstm*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
| GRU               | *gru*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py)
| MLP               | *mlp*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MLP.py)
| FCN               | *fcn*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py)
| ResNet            | *res_net*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py)
| LSTM-FCN          | *lstm_fcn*            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| GRU-FCN           | *gru_fcn*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| mWDN              | *mwdn*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/mWDN.py)
| TCN               | *tcn*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TCN.py)
| MLSTM-FCN         | *mlstm_fcn*           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py)
| InceptionTime     | *inception_time*      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py)
| InceptionTimePlus | *inception_time_plus* | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTimePlus.py)
| XcetptionTime      | *xception_time*       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XceptionTime.py)
| ResCNN            | *res_cnn*             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py)
| TabModel          | *tab_model*           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabModel.py)
| OmniScale         | *omni_scale*          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/OmniScaleCNN.py)
| TST               | *tst*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py)
| XCM               | *xcm*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py)

(C: Classifiction, R: Regression, F: Forecasting)

Here is a complete list of our supported models for computer vision:
| Model             | String                | Classification     | Hyperparameters    |
| ----------------- | --------------------- | ------------------ | ------------------ |
| Decision Tree     | *decision_tree*       | :heavy_check_mark: | [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) 
| Random Forest     | *random_forest*       | :heavy_check_mark: | [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
| AdaBoost          | *ada_boost*           | :heavy_check_mark: |[see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
| KNN               | *knn*                 | :heavy_check_mark: | [see](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
| GBDT              | *gbdt*                | :heavy_check_mark: | [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
| Gaussian NB       | *gaussian_nb*        | :heavy_check_mark: | [see](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
| SVM               | *svm*                 | :heavy_check_mark: | [see](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
| ResNet/ResNeta    | *res_net*             | :heavy_check_mark: | [see](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py)/[see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/resneta.py)
| SEResNet          | *se_res_net*          | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/seresnet.py)
| ResNeXt           | *res_next*            | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/resnext.py)
| AlexNet           | *alex_net*            | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/alexnet.py)
| VGG               | *vgg*                 | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/vgg.py)
| EfficientNet      | *efficient_net*       | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/efficientnet.py)
| WRN               | *wrn*                 | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/wrn.py)
| RegNet            | *reg_net*             | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/regnet.py)
| SCNet             | *sc_net*              | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/scnet.py)
| PANSNet           | *pnas_net*            | :heavy_check_mark: | [see](https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/pnasnet.py)
