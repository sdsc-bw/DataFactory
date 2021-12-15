# Model Selection
## What is model selection
There is a variety of models that can be used in machine learning like decision trees, random forests, neural networks...
According to the problem, some models fit better than others. If we have labeled data we can use supervised learning methods, whereas if we don't have the labels, we have to use other methods. Here a small overview:

![alt text](https://github.com/sdsc-bw/DataFactory/blob/develop/images/model_selection.png)

Also the models can perform vary on different tasks. For example, for a simple problem it makes sense to use a more simple model like a decision tree, because more complex models like neural networks can lead to overfitting. Whereas these complexe models perform better at non-linear problems.

In this Demo we want to present commonly used machine learning models and how they perform on different datasets.The models supported in this project are and how they can be adressed (C: Classifiction, R: Regression, F: Forecasting):

| Model             | String                | Classification     | Regression         | Forecasting              | Hyperparameters |
| ----------------- | --------------------- | ------------------ | ------------------ | ------------------------ | --------------- |
| Decision Tree     | *decision_tree*       | :heavy_check_mark: | :heavy_check_mark: | :x:                      | [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) [see](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
| Random Forest     | *random_forest*       | :heavy_check_mark: | :heavy_check_mark: | :x:                      |
| AdaBoost          | *adaboost*            | :heavy_check_mark: | :heavy_check_mark: | :x:                      |
| KNN               | *knn*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                      |
| GBDT              | *gbdt*                | :heavy_check_mark: | :heavy_check_mark: | :x:                      |
| GaussianNB        | *gaussian_nb*         | :heavy_check_mark: | :x:                | :x:                      |
| SVM               | *svm*                 | :heavy_check_mark: | :heavy_check_mark: | :x:                      |
| BayesianRidge     | *bayesian*            | :x:                | :heavy_check_mark: | :x:                      |
| InceptionTime     | *inception_time*      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:       |
| InceptionTimePlus | *inception_time_plus* | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:       |
| LSTM              | *lstm*                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:       |
| GRU               | *gru*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:       |
| MLP               | *MLP*                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:       |