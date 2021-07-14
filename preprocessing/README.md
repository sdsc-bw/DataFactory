#pre_processing
## What is preprecessing and why do we need it
The quality of the data directly determines quality of the prediction and generalization ability of the model. It involves many factors, including: accuracy, completeness, consistency, credibility. In the real world, the data we get may contain a lot of missing values, noise, or due to manual entry errors there may be outliers present, which is very unfavorable to the training of algorithmic models. The result of data preproessing is to process all kinds of dirty data in a corresponding way to get standard, clean and continuous data, which can be used for data statistics, data mining, etc.
## Common data pre-processing process 
### Data Import
There are differences in the extraction methods for data of different storage types. Common types include csv, bak, exel, json. Among them the most popular one is csv file.

When extracting csv files, there are three points to pay special attention to, they are.
- Delimiter used 
- header: row number of the columns name
- index_col: column used as row label

### Data Review
  Correctness review:
    - data inconsistency: e.g., birthday appears in name feature, age is inconsisten with birthday
    - data correctness: The data received is different from the data described by the customer: e.g.,asking for acceleration data, but RPM received; asking for monday's data, but Friday's data received.

  Usability review:
    - Mainly for uselessness(no contribution of the target), duplication, redundancy(data from backup sensor) and unmanageable errors in the data

The above errors contain a large number of subjective factors, so there is no way to be automatically identified by machine learning methods, and researchers are required to review specific data 

### Data Cleaning
The main idea of data cleaning is to "clean" data by filling in missing values, smoothing noisy data, smoothing or removing outliers, and resolving data inconsistencies. If customers think the data is dirty, they are less likely to trust the results based on this data i.e., the output is unreliable§.
The problems may vary from dataset to dataset. Some classic problems are summaried here.
#### Categorical features
Many popular models can't handle character feature, e.g., neural network

To handle these attributes, we generally convert the character data into numeric data via:
-  <a href = 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html'>LabelEncoder</a> or 
- <a href= 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html'>OnehotEncoder</a>

#### Datetime feature
Date data is usually defaulted to object type when the data is read. Direct encoding will mask the time information, in this case, we need to extract the time information by some other methods:
- <a href = 'https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html'>pd.to_datetime</a>
- <a href = 'https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_timestamp.html'>pd.to_timestamp</a>

####  NA value
NA value, or missing value is the missing value in the data. When the data with messing value is loaded, the missing value generally appears as NA. Missing values in general can be classified as follows：
- MAR(Missing at random): Random missing means that the probability of missing data is not related to the missing data itself, but only to the partially observed data. That is, the missing data is not completely random and the missing data of that type depends on other complete variables
- MCAR(Missing complete at random): Missing data is completely random
- MNAR(Missing not at random): Missing data is related to the value of the incomplete variable itself, e.g.,Women usually do not want to reveal their age.

There are many reasons for missing values, some of the more classic ones include:
- Information is temporarily unavailable
- Data is not recorded, omitted or lost due to human factors, this is the main reason for missing data
- Data loss due to failure of data acquisition equipment, storage media, transmission media failure
- The cost of obtaining this information is too high
- Some objects have one or more attributes that are not available, e.g., the name of the spouse of an unmarried person

The best way to handle NA value is to analyze the cause of generation and deal with them one by one according to the cause. However, this method is time consuming when the amount of missing values is large, and as an alternative, the following methods are generally used to handle missing values.
- delete feature: If a feature has a relatively high missing rate (80%) and is of low importance, you can simply delete the feature.
- delete record: Not recommended, unless most of the values of the record are na
- filling:
  - filling manually
  - filling dummy: Filling with special value, e.g., 0, 'other'
  - filling statistically: Filling in missing values with statistical values, e.g., mean, max, min, mode 
  - filling interpolation: including random interpolation, multiple differential interpolation, thermal platform interpolation, Lagrangian interpolation, Newton interpolation, etc.
  - filling model: Prediction of missing data using models such as K-nearest neighbor, regression, Bayesian, random forest, and decision tree. 

#### Outliers
Outliers are the norm of data distribution, and data that is outside a specific distribution area or range is usually defined as outlier(anomalous or noisy). There are two types of anomalies: 
- pseudo outliers, which are generated due to specific business operation actions and are a normal response to the state of the business, rather than anomalies in the data itself; 
- true outliers, which are not generated by specific business operations, but by anomalies in the distribution of the data itself, i.e., outliers. i.e., outliers. 

The main methods for detecting outliers are as follows.
- 3 sigma/IQR(Interquartile range:0.75 - 0.25)
- Median absolute deviation: |X-median(X)|/|median(|X-median(X)|)| < threshold
- Distance based 
- Density based
- Clustering based

After finding the outliers, we can handle them with following method:
    - Consider whether to delete records based on the number and impact of outliers ->  more information loss
    - log scale
    - treat as na value
    - ignore and select a model that is more robust to outliears, such as decision tree

#### Noise
Noise is the random error and variance of the variables and is the error between the observed and ground true. 

- The usual treatments include: the data are subjected to The data is divided into boxes, equal frequency or equal width, and then the mean, median or boundary value of each box (different data distribution The usual approach is to replace all the numbers in the box with the mean, median, or boundary value of each box (different data distributions, different treatment), which serves to smooth the data. 
- Another approach is to build a regression model of this variable and predictor variables, and inverse solve for an approximation of the independent variable based on the regression coefficients and predictor variables.
- Filtering
### Data balancing 
When dealing with the classification task, the number of instances for each classification varies, which tends to cause the model to favor one class in learning, thus affecting the performance of the model.

A similar problem exists in regression problems, where more input attributes or clusters are used to classify the data, and if the number of instances of each class is very different, the learning of the model will also be affected.
- Upsampling or Downsampling randomly
- Upsampling or Downsampling with <a href = 'https://imbalanced-learn.org/stable/references/index.html#api'>imbalanced-learn</a>
  - SMOTE
  - TomekLinks
### Data Transform
- normalization：mimmax,zscore, log, Box-Cox 
- Discretization: equal frequency or width, clustering
- Sparsification: onehot
### Feature engineering
