import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import sys

from abc import ABCMeta, abstractmethod
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import * 
from scipy.stats import iqr
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, AdaBoostClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, SVR
if 'autosklearn' in sys.modules:
    import autosklearn.classification # requires linux or colab
import hyperopt
from hyperopt import fmin, tpe, rand, hp, SparkTrials, STATUS_OK, Trials
import mlflow
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from transforms import UnaryOpt, BinaryOpt, MultiOpt
import operator
from tsai.all import *
computer_setup()

import util_codes as util
import transforms as tfd
import search_space as ss

class DataFactory:
    def __init__(self, threshold: float = .01) -> None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # hyper parameters
        self.threshold = threshold
        # operators
        self.opts_unary = self.load_opts(typ = 'unary')
        self.opts_binary = self.load_opts(typ = 'binary')
        self.opts_multi = self.load_opts(typ = 'multi')
        self.opts_cla_supervised = self.load_opts(typ= 'cla')
        self.opts_reg_supervised = self.load_opts(typ= 'reg')
        
        self.X = None
        self.y = None
        
        
    def pipeline(self, fn: str) -> Tuple[list, list]:
        """Loads data, computes baseline, generates meta features, tests transforamtion and wraps results.
        
        Keyword arguments:
        fn -- path to data
        """
        self.logger.info('='*60)
        self.logger.info(f'Extract trainings data from: {fn}')
        self.logger.info('='*60)
        self.logger.info('Start to load data, preprocessing and calculate baseline...')
        dat, target, baseline = self.load_data(fn)
        self.logger.info(f'...End of loading data: shape {dat.shape}, baseline {baseline}')
        cols = dat.columns
        
    
    def apply_binary_transformations_for_series(self, value1: pd.Series, value2: pd.Series) -> pd.DataFrame:
        values = []
        self.logger.info(f'Start to apply binary transformtions to series {value1.name} and series {value2.name}...')
        for key in self.opts_binary.keys():
            self.logger.info(f'Applying transformation: {self.opts_binary[key].name}')
            tmp_value = self.opts_binary[key].fit(value1, value2)
            values.append(tmp_value)
        self.logger.info(f'...End with transformation')
        return pd.concat(values, axis = 1)
            
    def apply_multiple_transformations_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = []
        self.logger.info(f'Start to apply multi transformtions to dataframe...')
        for key in self.opts_multi.keys():
            self.logger.info(f'Applying transformation: {self.opts_multi[key].name}')
            tmp_df = self.opts_multi[key].fit(df)
            dfs.append(tmp_df)
        self.logger.info(f'...End with transformation')
        return pd.concat(dfs, axis = 1)
    
    def apply_supervised_transformations_for_dataframe(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                       y_train: pd.Series, y_test: pd.Series, art = 'C') -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs_train, dfs_test = [], []
        self.logger.info(f'Start to apply supervised transformtions to dataframe...')
        opts = self.opts_cla_supervised if art == 'C' else self.opts_reg_supervised
        for key in opts.keys():
            self.logger.info(f'Applying transformation: {opts[key].name}')
            tmp_train, tmp_test = opts[key].fit(X_train, X_test, y_train, y_test)
            dfs_train.append(tmp_train)
            dfs_test.append(tmp_test)
        self.logger.info(f'...End with transformation')
        return pd.concat(dfs_train, axis = 1), pd.concat(dfs_test, axis = 1)
    
    def apply_unary_transformations_to_series(self, value: pd.Series) -> pd.DataFrame:
        values = []
        self.logger.info(f'Start to apply unary transformtions to series: {value.name}...')
        for key in self.opts_unary.keys():
            self.logger.info(f'Applying transformation: {self.opts_unary[key].name}')
            tmp_value = self.opts_unary[key].fit(value)
            values.append(tmp_value)
        self.logger.info(f'...End with transformation')
        return pd.concat(values, axis = 1)
    
    def categorical_feature_encoding(self, dfx: pd.DataFrame, dfy: pd.Series = None, k_term: bool = True):
        """Categorical feature encoding
        
        Keyword arguments:
        dfx -- data
        dfy -- labels
        k_term -- whether k-terms should be added as columns
        Output:
        dat_new -- encoded data
        out_y -- encoded labels (optional)
        """
        self.logger.info('Start to transform the categorical columns...')
        # replace original indeices with default ones
        if dfy is not None:
            out_y = dfy
            if dfy.dtype == 'O':
                self.logger.info('Start with label encoding of the target...')
                out_y = pd.Series(LabelEncoder().fit_transform(dfy), index = dfy.index)
                self.logger.info('...End with Target encoding')
        dat_categ = dfx.select_dtypes(include=['object'])
        dat_numeric = dfx.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
        dat_kterm = pd.DataFrame(index = dfx.index)
        # get kterm of categ features
        if k_term:
            self.logger.info('K-term function is activated, try to extract the k-term for each object columns')
            for i in dat_categ.columns:
                # get k-term feature
                tmp = dfx[i].value_counts()
                dat_kterm[i + '_kterm'] = dfx[i].map(lambda x: tmp[x] if x in tmp.index else 0)
            self.logger.info('...End with k-term feature extraction')
        # onehot encoding and label encoding
        dat_categ_onehot = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values < 8]
        dat_categ_label = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values >= 8]
        flag_onehot = False
        flag_label = False
        oe = OneHotEncoder(drop='first')
        # oe
        if dat_categ_onehot.shape[1] > 0:
            self.logger.info('Start with one-hot encoding of the following categoric features: %s...' %
                        (str(dat_categ_onehot.columns.to_list())))
            dat_onehot = pd.DataFrame(oe.fit_transform(dat_categ_onehot.astype(str)).toarray(),
                                      columns=oe.get_feature_names(dat_categ_onehot.columns))
            self.logger.info('...End with one-hot encoding')
            flag_onehot = True
        else:
            dat_onehot = None
        # le
        if dat_categ_label.shape[1] > 0:
            self.logger.info('Start label encoding of the following categoric features: %s...' %
                        (str(dat_categ_label.columns.to_list())))
            dat_categ_label = dat_categ_label.fillna('NULL')
            dat_label = pd.DataFrame(columns=dat_categ_label.columns)
            for i in dat_categ_label.columns:
                dat_label[i] = LabelEncoder().fit_transform(dat_categ_label[i].astype(str))
            flag_label = True
            self.logger.info('...End with label encoding')
        else:
            dat_label = None
        # scaling
        # combine
        dat_new = pd.DataFrame()
        if flag_onehot and flag_label:
            dat_new = pd.concat([dat_numeric, dat_onehot, dat_label], axis=1)
        elif flag_onehot:
            dat_new = pd.concat([dat_numeric, dat_onehot], axis=1)
        elif flag_label:
            dat_new = pd.concat([dat_numeric, dat_label], axis=1)
        else:
            dat_new = dat_numeric
        if k_term:
            dat_new = pd.concat([dat_new, dat_kterm], axis = 1)
        self.logger.info('...End with categorical feature transformation')
        if dfy is not None:
            return dat_new, out_y
        else:
            return dat_new
    
    def clean_dat(self, dat: pd.DataFrame, strategy: str = 'model') -> pd.DataFrame:
        """Clean dataset of INF- and NAN-values.
        
        Keyword arguments:
        dat -- dataframe
        strategy -- cleaning strategy, should be in ['model', 'mean', 'median', 'most_frequent', 'constant']
        Output:
        dat -- cleaned dataframe
        """
        if dat.empty:
            return dat
        self.logger.info('Start to clean the given dataframe...')
        self.logger.info('Number of INF- and NAN-values are: (%d, %d)' % ((dat == np.inf).sum().sum(), dat.isna().sum().sum()))
        self.logger.info('Set type to float32 at first && deal with INF')
        dat = dat.astype(np.float32)
        dat = dat.replace([np.inf, -np.inf], np.nan)
        self.logger.info('Remove columns with half of NAN-values')
        dat = dat.dropna(axis=1, thresh=dat.shape[0] * .5)
        self.logger.info('Remove constant columns')
        dat = dat.loc[:, (dat != dat.iloc[0]).any()]

        if dat.isna().sum().sum() > 0:
            self.logger.info('Start to fill the columns with NAN-values...')
            # imp = IterativeImputer(max_iter=10, random_state=0)
            if strategy == 'model':
                imp = IterativeImputer(max_iter=10, random_state=0)
            elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
                imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
            else:
                self.logger.warn(f'Unrecognized strategy {strategy}. Use mean instead')
            # dat = dat.fillna(dat.mean())
            tmp = imp.fit_transform(dat)
            if tmp.shape[1] != dat.shape[1]:
                self.logger.warn(f'Error appeared while fitting. Use constant filling instead')
                tmp = dat.fillna(0)
            dat = pd.DataFrame(tmp, columns=dat.columns, index=dat.index)
        #logger.info('Remove rows with any nan in the end')
        #dat = dat.dropna(axis=0, how='any')
        self.logger.info('...End with Data cleaning, number of INF- and NAN-values are now: (%d, %d)' 
                     % ((dat == np.inf).sum().sum(), dat.isna().sum().sum()))
        #dat = dat.reset_index(drop=True)
        return dat

    def evaluate(self, dat: pd.DataFrame, target: pd.Series, cv: int = 5, mtype = 'C') -> Tuple[float, float]:
        """Evaluates a dataset with random forests and f1-scores.
        
        Keyword arguments:
        dat -- dataframe
        target -- labels
        cv -- number of random forests
        mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
        Output:
        mean -- mean of scores
        var -- variance of scores
        """
        scores = []
        for i in range(cv):
            X_train, X_test, y_train, y_test = train_test_split(dat, target, random_state = i)
            if mtype == 'C':
                model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                score = f1_score(y_test, predict, average='weighted')
            elif mtype == 'R':
                model = RandomForestRegressor(random_state = self.rf_seed)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                score = 1 - self._relative_absolute_error(predict, y_test)
            else:
                self.logger.error('Unknown type of task')
            scores.append(score)
        return np.mean(scores), np.std(scores)
    
    def extract_date_information(self, value:pd.Series)->pd.DataFrame:
        self.logger.info(f'Start to extract datetime information from: {value.name}...')
        tmp = pd.to_datetime(value)
        out = tmp.apply(lambda x: pd.Series([x.year, x.month, x.day, x.dayofyear, x.dayofweek,
                                 x.hour, x.minute, x.second, x.microsecond], 
                                 index = [value.name +'_'+ i for i in ['year', 'month', 'day', 'dayofyear', 'dayofweek', 'hour', 'minute', 'second','microsecond']]))
        self.logger.info('...End with date time information extraction')
        return out
    
    def load_data(self, fn: str, sep: str=',') -> Tuple[pd.DataFrame, pd.Series, float]:
        df = load_data(fn, logger = self.logger)
        dat, target = df.iloc[:, :-1], df.iloc[:, -1]
        baseline, _ = self._evaluate(dat, target)
        return dat, target, baseline
        
    def load_opts(self, typ: str) -> Dict[str, Union[UnaryOpt, BinaryOpt]]:
        if typ == 'unary':
            operators = {'abs': tfd.Abs(), 'cos': tfd.Cos(), 'degree': tfd.Degree(), 'exp': tfd.Exp(), 'ln': tfd.Ln(), 'negative': tfd.Negative(), 
                        'radian': tfd.Radian(), 'reciprocal': tfd.Reciprocal(), 'sin': tfd.Sin(), 'sigmoid': tfd.Sigmoid(), 'square': tfd.Square(),
                        'tanh': tfd.Tanh(), 'relu': tfd.Relu(), 'sqrt': tfd.Sqrt(), 'binning': tfd.Binning(),
                        'ktermfreq': tfd.KTermFreq()}
        elif typ == 'binary':
            operators = {'div': tfd.Div(), 'minus': tfd.Minus(), 'add': tfd.Add(), 'product': tfd.Product()}
        elif typ == 'multi':
            operators = {'clustering': tfd.Clustering(), 'diff': tfd.Diff(), 'minmaxnorm': tfd.Minmaxnorm(),
                         'winagg': tfd.WinAgg(), 'zscore': tfd.Zscore(), 'nominalExpansion': tfd.NominalExpansion(),
                         'isomap': tfd.IsoMap(), 'leakyinfosvr': tfd.LeakyInfoSVR(), 'kernelAppRBF': tfd.KernelApproxRBF()}
        elif typ == 'cla':
            operators = {'dfCla': tfd.DecisionTreeClassifierTransform(), 'mlpCla': tfd.MLPClassifierTransform(),
                         'knCla': tfd.NearestNeighborsClassifierTransform(), 'svCla': tfd.SVCTransform(), 
                         'gdwCla': tfd.GauDotClassifierTransform(), 'geCla': tfd.GauExpClassifierTransform(),
                         'grbfCla': tfd.GauRBFClassifierTransform(), 'rfCla': tfd.RandomForestClassifierTransform(),
                         'xgbCla': tfd.XGBClassifierTransform()}
        elif typ == 'reg':
            operators = {'dtReg': tfd.DecisionTreeRegressorTransform(), 'liReg': tfd.LinearRegressorTransform(),
                         'mlpReg': tfd.MLPRegressorTransform(), 'knReg': tfd.NearestNeighborsRegressorTransform(),
                         'svReg': tfd.SVRTransform(), 'gdwReg': tfd.GauDotWhiteRegressorTransform(),
                         'geReg': tfd.GauExpRegressorTransform(), 'grbfReg': tfd.GauRBFRegressorTransform(),
                         'rfReg': tfd.RandomForestRegressorTransform(), 'xgbReg': tfd.XGBRegressorTransform()}
        return operators
    
        
    def outlier_detection_dataframe(self, df: pd.DataFrame, strategy: str = 'density') -> pd.Series:
        """Outlier detection of a given dataframe.
        
        Keyword arguments:
        value -- dataframe
        Output:
        out -- outlier of given dataframe
        """
        self.logger.info(f'Start to detect outlier for the whole data set with strategy: {strategy}...')
        if strategy == 'high_dimension':
            out = self._outlier_detection_high_dimension(df)
        elif strategy == 'density':
            out = self._outlier_detection_density(df)
        else:
            self.logger.info('Unrecognized strategy. Use density-based strategy instead')
            out = self._outlier_detection_density(df)
        self.logger.info(f'...End with outlier detection, {out.sum()} outliers found')
        return out

    def outlier_detection_feature(self, value: pd.Series) -> pd.Series:
        """Outlier detection of a given feature.
        
        Keyword arguments:
        value -- dataframe with values of a given feature
        Output:
        out -- outlier of given feature
        """
        self.logger.info(f'Start to detect outlier for given feature {value.name} with 3 IQR strategy...')
        v_iqr = iqr(value)
        v_mean = value.mean()
        ceiling = v_mean + 3*v_iqr
        floor = v_mean - 3*v_iqr
        out = value.map(lambda x: x>ceiling or x<floor)
        self.logger.info(f'...End with outlier detection, {out.sum()} outliers found')
        return out

    def _outlier_detection_high_dimension(self, df: pd.DataFrame) -> pd.Series:
        """High dimension outlier detection of a given dataframe with a random forest.
        
        Keyword arguments:
        value -- dataframe
        Output:
        out -- outlier of given dataframe
        """
        clf = IsolationForest(n_estimators=20, warm_start=True)
        out = pd.Series(clf.fit_predict(df), index = df.index)
        out = out.map(lambda x: x == -1)
        return out

    def _outlier_detection_density(self, df: pd.DataFrame) -> pd.Series:
        """Density-based outlier detection of a given dataframe.
        
        Keyword arguments:
        value -- dataframe
        Output:
        out -- outlier of given dataframe
        """
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        out = pd.Series(clf.fit_predict(df), index = df.index)
        out = out.map(lambda x: x == -1)
        return out

    def plot_linear_for_columns_in_df(self, df: pd.DataFrame, step: int, cols: list, save_path = None, id = None):
        """Creates a linear plot of the given columns in the dataframe. Saves plot at given path.
        
        Keyword arguments:
        df -- dataframe
        step -- plot the value every ||step|| items
        cols -- columns that should be plotted
        save_path -- path where to save the plot
        id -- ID of the plot
        """
        plt.figure(figsize=(20,6))
        df = df.reset_index(drop = True)
        for i in cols:
            tmp = df.loc[np.arange(0, len(df), step), i]
            plt.plot(tmp.index, tmp, label = i) # /1000
        plt.xlabel('Second')
        plt.legend()
        plt.title(str(id))
        plt.tick_params(axis='x',labelsize=18)
        plt.tick_params(axis='y',labelsize=18)
        plt.legend(prop={'size': 16})
        if save_path:
            save_path = save_path+'_step_'+str(step)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(save_path+ '/' +'_'.join(cols)+ '_' + str(id), transparent = True)
        
    def plot_density_for_each_column_in_df(self, df: pd.DataFrame, save_path: str = None, id = None):
        """Creates a density plots of each column in the dataframe. Saves plot at given path.
        
        Keyword arguments:
        df -- dataframe
        save_path -- path where to save the plot
        id -- ID of the plot
        """
        for i in df.columns:
            plt.figure(figsize=(20,6))
            sns.kdeplot(df[i])
            #plt.xlabel('Second')
            plt.legend()
            plt.title(str(i))
            plt.tick_params(axis='x',labelsize=18)
            plt.tick_params(axis='y',labelsize=18)
            plt.legend(prop={'size': 16})
        if save_path:
            save_path = save_path+'_step_'+str(step)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(save_path+ '/' +'_'.join(cols)+ '_' + str(id), transparent = True)
    
    def sampling_up(self, dfx: pd.DataFrame, dfy: pd.Series, strategy: str = 'SMOTE', random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Samples down dataset based on given sampling strategy.
        Keyword arguments:
        dfx -- data
        dfy -- labels
        strategy -- sampling strategy, should be in ['smote', 'random', 'borderline', 'adasyn', 'kmeanssmote']
        random_state -- controlls the randomization of the algorithm
        
        Output:
        res_x -- up sampled data
        res_y -- up sampled labels
        """
        self.logger.info(f'Start to apply upsampling strategy: {strategy}...')
        if strategy == 'smote':
            usa = SMOTE(sampling_strategy='auto', random_state=random_state)
        elif strategy == 'random':
            usa = RandomOverSampler(sampling_strategy='auto', random_state=random_state)
        elif strategy == 'borderline':
            usa = BorderlineSMOTE(sampling_strategy='auto', random_state=random_state)
        elif strategy == 'adasyn':
            usa = ADASYN(sampling_strategy='auto', random_state=random_state)
        elif strategy == 'kmeanssmote':
            usa = KMeansSMOTE(sampling_strategy='auto', random_state=random_state)
        else:
            logger.warn('Unrecognized upsampling strategy. Use SMOTE instead')
            usa = SMOTE(sampling_strategy='auto', random_state=random_state)
        res_x, res_y = usa.fit_resample(dfx, dfy)
        if type(res_x) == np.ndarray:
            res_x = pd.DataFrame(res_x, columns = dfx.columns)
            res_y = pd.Series(res_y, name = dfy.name)
        self.logger.info('...End with upsampling')
        return res_x, res_y

    def sampling_down(self, dfx: pd.DataFrame, dfy: pd.Series, strategy: str = 'SMOTE', random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Samples down dataset based on given sampling strategy.
        Keyword arguments:
        dfx -- data
        dfy -- labels
        strategy -- sampling strategy, should be in ['cluster', 'random', 'nearmiss1', 'nearmiss2', 'nearmiss3', 'tomek', 'enn', 'repeatenn', 'allknn', 'condensednn']
        random_state -- controlls the randomization of the algorithm
        
        Output:
        res_x -- down sampled data
        res_y -- down sampled labels
        """
        self.logger.info(f'Start to apply downsampling strategy: {strategy}...')
        if strategy == 'cluster':
            dsa = ClusterCentroids(sampling_strategy = 'auto', random_state = random_state)
        elif strategy == 'random':
            dsa = RandomUnderSampler(sampling_strategy = 'auto', random_state = random_state)
        elif strategy == 'nearmiss1':
            dsa = NearMiss(sampling_strategy = 'auto', version = 1)
        elif strategy == 'nearmiss2':
            dsa = NearMiss(sampling_strategy = 'auto', version = 2)
        elif strategy == 'nearmiss3':
            dsa = NearMiss(sampling_strategy = 'auto', version = 3)
        elif strategy == 'tomek':
            dsa = TomekLinks(sampling_strategy = 'auto')
        elif strategy == 'enn':
            dsa = EditedNearestNeighbours(sampling_strategy = 'auto')
        elif strategy == 'repeatenn':
            dsa = RepeatedEditedNearestNeighbours(sampling_strategy = 'auto')
        elif strategy == 'allknn':
            dsa = AllKNN(sampling_strategy = 'auto')
        elif strategy == 'condensednn':
            dsa = CondensedNearestNeighbour(sampling_strategy = 'auto', random_state = random_state)
        else:
            self.logger.warn('Unrecognized downsampling strategy. Use TOMEK instead')
            dsa = TomekLinks(sampling_strategy = 'auto')
        res_x, res_y = dsa.fit_resample(dfx, dfy)
        if type(res_x) == np.ndarray:
            res_x = pd.DataFrame(res_x, columns = dfx.columns)
            res_y = pd.Series(res_y, name = dfy.name)
        self.logger.info('...End with downsampling')
        return res_x, res_y

    def sampling_combine(self, dfx: pd.DataFrame, dfy: pd.Series, strategy: str = 'SMOTE', random_state: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Samples dataset based on given sampling strategy.
        Keyword arguments:
        dfx -- data
        dfy -- labels
        strategy -- sampling strategy, should be in ['smoteenn', 'smotetomek']
        
        Output:
        X_train -- trainings data
        X_test -- test data
        y_train -- trainings label
        y_test -- test label
        """
        self.logger.info(f'Start to apply combine sampling strategy: {strategy}...')
        if strategy == 'smoteenn':
            csa = SMOTEENN(sampling_strategy = 'auto', random_state = random_state)
        elif strategy == 'smotetomek':
            csa = SMOTETomek(sampling_strategy = 'auto', random_state = random_state)
        else:
            self.logger.warn('Unrecognized downsampling strategy... Use SMOTEENN instead')
            csa = SMOTEENN(sampling_strategy = 'auto', random_state = random_state)
        res_x, res_y = csa.fit_resample(dfx, dfy)
        if type(res_x) == np.ndarray:
            res_x = pd.DataFrame(res_x, columns = dfx.columns)
            res_y = pd.Series(res_y, name = dfy.name)
        self.logger.info('...End with combine sampling')
        return res_x, res_y

    def preprocess(self, dat: pd.DataFrame, y_col: str) -> Tuple[np.array, np.array]:
        """Preprocesses data.
        Keyword arguments:
        dat -- dataframe with dataset
        y_col -- name of target column
        
        Output:
        dfx -- preprocessed data
        dfy -- preprocessed labels
        """
        self.logger.info(f'Remove columns with NAN-values of target feature: {y_col}')
        df = dat.dropna(subset=[y_col])
        x_columns = list(df.columns)
        x_columns.remove(y_col)
        dfx = df[x_columns]
        dfy = df[y_col]
        dfx, dfy = self.categorical_feature_encoding(dfx, dfy, k_term=False)
        dfx = self.clean_dat(dfx)
        return dfx, dfy

    def preprocess_and_split(self, dat: pd.DataFrame, y_col: str) -> Tuple[np.array, np.array, np.array, np.array]:
        """Preprocesses data and splits data into trainingsset and testset.
        Keyword arguments:
        dat -- dataframe with dataset
        y_col -- name of target column
        
        Output:
        X_train -- preprocessed training data
        X_test -- preprocessed test data
        y_train -- preprocessed training label
        y_test -- preprocessed test label
        """
        dfx, dfy = self.preprocess(self, dat, y_col)
        X_train, X_test, y_train, y_test = train_test_split(dfx, dfy)
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, dfx: pd.DataFrame, dfy: pd.Series = None, strategy: str = 'grid', cv: int = 5, model='decision_tree', mtype='C', params: Dict = None):
        """Trains and evaluates a given model.
        
        Keyword arguments:
        dfx -- data
        dfy -- labels
        strategy -- search strategy of hyperparameters should be in ['grid', 'random']
        cv -- number of model instances
        model -- model should be in ['decision_tree', 'random_forest', 'adaboost', 'knn', 'gbdt', 'gaussian_nb' 'svm', 'bayesian']
        mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
        Output:
        best_model -- the model with the highest score
        score -- score of the best model
        """
        self.logger.info(f'Start search for best parameters of: {model}...')
        X_train, X_test, y_train, y_test = train_test_split(dfx, dfy)
        if params is None:
            params = ss.get_std_search_space(model)
            
        m = self._get_model(mtype, params['model'])
          
        if strategy == 'grid':
            grid = GridSearchCV(m, param_grid=params, refit=True, cv=cv, verbose=verbose, n_jobs=-1)
        elif strategy == 'random':
            grid = RandomizedSearchCV(m, param_distributions=params, refit=True, cv=cv, verbose=verbose, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        score = _get_score(y_pred, y_test, mtype)
        
        self.logger.info(f'...End search')
        self.logger.info(f'Best parameters are: {grid.best_params_}')
        return best_model, score
    
    
    def train_and_evaluate_network(self, dfx: pd.DataFrame, dfy: pd.Series, model='inception_time', mtype='C', epochs: int=25, lr_max=1e-3, loss=None, optimizer='adam', params: Dict=dict(), transforms=None, batch_size=[64, 128], splits=None, metrics=['accuracy']):
        arch = self._get_network(model)
        loss = util.get_loss(loss)
        optimizer = util.get_optimizer(optimizer)
        metrics = util.get_metrics(metrics)
        
        self.logger.info(f'Start training of: {model}...')
        
        if mtype=='C':
            learn = TSClassifier(dfx, dfy, splits=splits, bs=batch_size, batch_tfms=transforms, arch=arch, arch_config=params, loss_func=loss, opt_func=optimizer, metrics=metrics)
        elif mtype=='R':
            learn = TSRegressor(dfx, dfy, splits=splits, bs=batch_size, batch_tfms=transforms, arch=arch, arch_config=params, loss_func=loss, opt_func=optimizer, metrics=metrics)
        elif mtype=='F':
            learn = TSForecaster(dfx, dfy, splits=splits, bs=batch_size, batch_tfms=transforms, arch=arch, arch_config=params, loss_func=loss, opt_func=optimizer, metrics=metrics)
        else:
            self.logger.error('Unknown type of model')
            
        learn.fit_one_cycle(epochs, lr_max=lr_max)
        learn.plot_metrics()
        
        self.logger.info(f'...End training')
        
        return learn
    
    def finetune(self, dfx: pd.DataFrame, dfy: pd.Series=None, cv: int=5, method: str='hyperopt', models:list=['decision_tree'], mtype='C', params: Dict=dict()):
        """Finetunes one or multiple models according to the given method.
        
        Keyword arguments:
        dfx -- data
        dfy -- labels
        cv -- number of model instances
        method -- finetuning method should be in ['auto_sklearn', 'hyperopt', 'sklearn']
        models -- models consist of ['decision_tree', 'random_forest', 'adaboost', 'knn', 'gbdt', 'gaussian_nb' 'svm', 'bayesian']
        mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
        params -- list of dictionaries with parameter to try out
        Output:
        best_model -- the model with the highest score
        best_score -- score of the best model
        """
        self.logger.info(f'Start finetuning...')
        if method == 'auto_sklearn':
            best_model, best_score = self._finetune_auto_sklearn(dfx, dfy, mtype=mtype, params=params)
        elif method == 'hyperopt':
            best_model, best_score, best_params = self._finetune_hyperopt(dfx, dfy, params, models, cv, mtype)
            m_str = util.model_to_string(best_model)
            self.logger.info(f'...End finetuning')
            del best_params['cv']
            del best_params['model']
            del best_params['type']
            self.logger.info(f'Best model is: {m_str} with parameters: {best_params}')
        elif method == 'native':
            ms = dict()
            strategy = params.get('strategy', 'random')
            if 'strategy' in params:
                del params['strategy']
            for m in models:
                m_params = ss.get_sklearn_search_space(m, mtype, params)
                ms[m] = self._finetune_native(dfx, dfy, cv=cv, model=m, mtype=mtype, params=m_params.copy(), strategy=strategy) 
            m_str = max(ms.items(), key=operator.itemgetter(0))[0]
            best_model, best_score, best_params = ms[m_str]
            self.logger.info(f'...End finetuning')
            self.logger.info(f'Best model is: {m_str} with parameters: {best_params}')
        return best_model, best_score
    
    def _finetune_auto_sklearn(self, dfx: pd.DataFrame, dfy: pd.Series=None, mtype: str='C', params: Dict=dict()):
        # TODO maybe delet params
        X_train, X_test, y_train, y_test = train_test_split(dfx, dfy)
        if mtype == 'C':
            best_model = autosklearn.classification.AutoSklearnClassifier()
        elif mtype == 'R':
            best_model = autosklearn.classification.AutoSklearnRegressor()
        else:
            self.logger.error('Unknown type of model')
        best_model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_test)        
        best_score = self._get_score(y_pred, y_test, mtype)
        return best_model, best_score
       
    
    def _finetune_native(self, dfx: pd.DataFrame, dfy: pd.Series=None, cv: int=5, model='decison_tree', mtype='C', params: Dict=dict(), strategy='random'):
        """Finetunes a given model with sklearn.
        
        Keyword arguments:
        dfx -- data
        dfy -- labels
        cv -- number of model instances
        model -- model should be in ['decision_tree', 'random_forest', 'adaboost', 'knn', 'gbdt', 'gaussian_nb' 'svm', 'bayesian']
        mtype -- type of the model, should be in ['C', 'R'] (C: Classifier, R: Regressor)
        params -- parameter to try out
        Output:
        best_model -- the model with the highest score
        best_score -- score of the best model
        """
        self.logger.info(f'Start search for best parameters of: {model}...')
        if util.get_library(model) == 'sklearn':
            X_train, X_test, y_train, y_test = train_test_split(dfx, dfy)
            model = self._get_model(mtype, model)
            if strategy == 'grid':
                search = GridSearchCV(model, param_grid=params, refit=True, cv=cv, n_jobs=-1)
            elif strategy == 'random':
                search = RandomizedSearchCV(model, param_distributions=params, refit=True, cv=cv, n_jobs=-1)
            else:
                self.logger.error('Unknown strategy')

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)        
            best_score = self._get_score(y_pred, y_test, mtype)
            best_params = best_model.get_params()
        elif util.get_library(model) == 'tsai':
            tfms = params.get('transforms', None)
            splits = params.get('splits', None)
            bs = params.get('bs', None)
            arch_config = params.get(model, dict())
            epochs = params.get('epochs', 25)
            lr_max = params.get('lr_max', 1e-3)
            
            dsets = TSDatasets(dfx, dfy, tfms=tfms, splits=splits)
            dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=bs)
            
            model = self._get_network(model)
            # todo use f1 instead of accuracy
            learn = Learner(dls, model,  metrics=accuracy)
            learn.fit_one_cycle(epochs, lr_max)
            best_model = learn
            best_score = learn.recorder.values[-1][2]
            best_params = arch_config
        
        self.logger.info(f'...End search')
        self.logger.info(f'Best parameters are: {best_params} with score {best_score:.2f}')
        return best_model, best_score, best_params
    
    def _finetune_hyperopt(self, dfx: pd.DataFrame, dfy: pd.Series, params, models, cv, mtype):
        self.X = dfx
        self.y = dfy
        trials = Trials()
        strategy = params.get('strategy', 'random')
        if 'strategy' in params:
            del params['strategy']
            
        search_space = ss.get_hyperopt_search_space(models, mtype, cv, params)
        
        if strategy == 'parzen':
            algo = tpe.suggest
        elif strategy == 'random':
            algo = rand.suggest
        else:
            self.logger.error('Unknown strategy')
        
        with mlflow.start_run():
            best_result = fmin(
            fn=self._objective, 
            space=search_space,
            algo=algo,
            max_evals=32,
            trials=trials)
            
        best_params = hyperopt.space_eval(search_space, best_result)
        
        best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
        best_score = -trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
        
        return best_model, best_score, best_params   
    
    def _relative_absolute_error(self, pred, y):
        dis = abs((pred-y)).sum()
        dis2 = abs((y.mean() - y)).sum()
        if dis2 == 0 :
            return 1
        return dis/dis2
    
    def _get_score(self, y_pred, y_test, mtype):
        if mtype == 'C':
            score = f1_score(y_test, y_pred, average='weighted')
        elif mtype == 'R':
            score = 1 - self._relative_absolute_error(y_pred, y_test)
        else:
            self.logger.error('Unknown type of model')
        return score
            
    def _objective(self, params):
        model = params['model']
        mtype = params['type']
        cv = params['cv']
        del params['model']
        del params['type']
        del params['cv']
        
        
        if util.get_library(model) == 'tsai':
            params = params.copy()
            epochs = params.get('epochs', 25)
            lr_max = params.get('lr_max', 1e-3)
            
            if 'epochs' in params:
                del params['epochs']
            if 'lr_max' in params:
                del params['lr_max']
        
        m = self._get_model(mtype, model, params)
        
        # TODO use f1 instead of accuracy
        if util.get_library(model) == 'sklearn': 
            score = cross_val_score(m, self.X, self.y, cv=cv, scoring='accuracy').mean()
            #score = cross_val_score(m, self.X, self.y, cv=cv, scoring='f1_weighted').mean()
        elif util.get_library(model) == 'tsai':
            score = self.cross_val_score_tsai(m, cv, epochs, lr_max).mean()
        return {'loss': -score, 'status': STATUS_OK, 'model': m}
        
    def cross_val_score_tsai(self, model, cv, epochs, lr_max):
        scores = np.zeros(cv)
        for i in range(cv):
            model.fit_one_cycle(epochs, lr_max=lr_max)
            # TODO how to use f1 instead of accuracy?
            scores[i] = model.recorder.values[-1][2]
        return scores
        
    def _get_model(self, mtype, model, params=dict()):
        if model == 'decision_tree':
            if mtype=='C':
                return tree.DecisionTreeClassifier(**params)
            elif mtype=='R':
                return tree.DecisionTreeRegressor(**params)
            else:
                self.logger.error('Unknown type of model')
        elif model == 'random_forest':              
            if mtype=='C':
                return RandomForestClassifier(**params)
            elif mtype=='R':
                return RandomForestRegressor(**params)
            else:
                self.logger.error('Unknown type of model')
        elif model == 'adaboost':               
            if mtype=='C':
                return AdaBoostClassifier(**params)
            elif mtype=='R':
                return AdaBoostRegressor(**params)
            else:
                self.logger.error('Unknown type of model')
        elif model == 'knn':              
            if mtype=='C':
                return KNeighborsClassifier(**params)
            elif mtype=='R':
                return KNeighborsRegressor(**params)
            else:
                self.logger.error('Unknown type of model')     
        elif model == 'gbdt':                
            if mtype=='C':
                return HistGradientBoostingClassifier(**params)
            elif mtype=='R':
                return HistGradientBoostingRegressor(**params)
            else:
                self.logger.error('Unknown type of model')                  
        elif model == 'gaussian_nb':
            if mtype=='C':
                return GaussianNB(**params)
            else:
                self.logger.error('Unknown type of model')
        elif model == 'svm':                
            if mtype=='C':
                return SVC(**params)
            elif mtype =='R':
                return SVR(**params)
            else:
                self.logger.error('Unknown type of model')  
        elif model == 'bayesian':      
            if mtype=='R':
                return linear_model.BayesianRidge(**params)
            else:
                self.logger.error('Unknown type of model')
        elif model == 'inception_time':
            ## TODO clean up loss + optimizer
            params['loss_func'] = util.get_loss(params.get('loss_func', 'cross_entropy'))
            params['opt_func'] = util.get_optimizer(params.get('opt_func', 'adam'))
            params['metrics'] = util.get_metrics(params.get('metrics', 'accuracy'))
            params['batch_tfms'] = util.get_transforms(params.get('batch_tfms', []))
            params['X'] = self.X
            params['y'] = self.y
            
            if mtype == 'C':
                return TSClassifier(**params)
            elif mtype == 'R':
                return TSRegressor(**params)
            elif mtype == 'F':
                return TSForecaster(**params)
            else:
                self.logger.error('Unknown type of model')
        elif mtype == 'C':    
            self.logger.info('Unrecognized classifier. Use decision tree instead')   
            return tree.DecisionTreeClassifier(**params)
        elif mtype == 'R':
            self.logger.info('Unrecognized regressor. Use decision tree instead')   
            return tree.DecisionTreeRegressor(**params)
        else:
            self.logger.error('Unknown type of model')
    
    def _get_network(self, model):
        if model == 'inception_time':
            return InceptionTime
        elif model == 'inception_time_plus':
            return InceptionTimePlus
        elif model == 'lstm':
            return LSTM
        elif model == 'gru':
            return GRU
        elif model == 'mlp':
            return MLP
        elif model == 'fcn':
            return FCN
        elif model == 'res_net':
            return ResNet
        elif model == 'lstm_fcn':
            return LSTM_FCN
        elif model == 'gru_fcn':
            return GRU_FCN
        elif model == 'mwdn':
            return mWDN
        elif model == 'tcn':
            return TCN
        elif model == 'xception_time':
            return XceptionTime
        elif model == 'res_cnn':
            return ResCNN
        elif model == 'tab_model':
            return TabModel
        elif model == 'omni_scale':
            return OmniScale
        elif model == 'tst':
            return TST
        elif model == 'xcm':
            return XCM
        elif model == 'mini_rocket':
            return MiniRocket
        else:
            self.logger.error('Unknown type of model') 
    
    def plot_model_comparison(self, xdat, ydat, ptype='plot', title=''):
        n_figures = len(ydat)
        fig, axs = plt.subplots(n_figures, constrained_layout=True, figsize=(10, 10))
        fig.suptitle(title)
        fig = plt.figure()

        if ptype == 'plot':
            for i in range(len(ydat)):
                if len(xdat) == 1:
                    axs[i].plot(xdat[0], ydat[i])
                else:
                    axs[i].plot(xdat[i], ydat[i])
        elif ptype == 'bar':
            for i in range(len(ydat)):
                if len(xdat) == 1:
                    axs[i].bar(xdat[0], ydat[i])
                    axs[i].tick_params(axis='x', labelrotation=45)
                    axs[i].set_ylim([0, 1.1])
                    axs[i].set_yticks(np.arange(0, 1.5, 0.25))
                else:
                    axs[i].bar(xdat[i], ydat[i])
                    axs[i].tick_params(axis='x', labelrotation=45)
                    axs[i].set_ylim([0, 1.1])
                    axs[i].set_yticks(np.arange(0, 1.5, 0.25))
        plt.show()