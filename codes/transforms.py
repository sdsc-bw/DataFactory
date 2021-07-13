from abc import ABCMeta, abstractmethod
import sys
import traceback
from numpy.core.numeric import NaN

from numpy.ma.core import append, reshape
from torch.utils import data

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind
from multiprocessing import cpu_count
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic, Exponentiation, RBF
from sklearn.kernel_approximation import RBFSampler
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, \
    QuantileTransformer, PowerTransformer
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb

from typing import cast, Any, Dict, List, Tuple, Optional, Union


class UnaryOpt(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, value: pd.Series) -> pd.Series:
        pass

class BinaryOpt(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, value1: pd.Series, value2: pd.Series) -> pd.Series:
        pass
    
class MultiOpt(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

class Transform(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass
    
def valid_col(col: pd.Series) -> bool:
    """ if there are inf, NaN, number out of range of float32, too many large/small numbers,  this new_col should be discarded"""
    if (col.isna().any() or
        (col == np.inf).any() or
        (col > np.finfo(np.float32).max).any() or
        (col < np.finfo(np.float32).min).any() or
        (abs(col - 0.0) < 0.0001).sum() / len(col) > 0.8):
        return False
    return True

## type A: unary

class Abs(UnaryOpt):
    def __init__(self):
        super(Abs, self).__init__()
        self.name = 'abs'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.abs()
        out.name = 'abs('+out.name+')'
        return out

class Adde(UnaryOpt):
    def __init__(self):
        super(Adde, self).__init__()
        self.name = 'adde'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value + np.e
        out.name = out.name+'+e'
        return out

class Binning(Transform):
    def __init__(self, k:int = 8):
        """
        clustering for each feature, need predefined k
        only _execute when there are less than 100 cols
        """
        super(Binning, self).__init__()
        self.name = 'binning'
        self.type = 1
        self.condition = 'data cleaning needed'
        self.k = k

    def fit(self, value: pd.Series) -> pd.Series:
        # do clustering for each column
        cluster = KMeans(n_clusters = self.k).fit(pd.DataFrame(value))
        out = pd.Series(cluster.predict(pd.DataFrame(value)))
        out.name = 'binning(' + str(value.name)+')'
        return out

class Cos(UnaryOpt):
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.cos(value)
        out.name = 'cos('+out.name+')'
        return out

class Degree(UnaryOpt):
    def __init__(self):
        super(Degree, self).__init__()
        self.name = 'degree'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.degrees(value)
        out.name = 'degree('+out.name+')'
        return out


class Exp(UnaryOpt):
    """
    only do exp when all the x <= 1
    """

    def __init__(self):
        super(Exp, self).__init__()
        self.name = 'exp'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.exp(value)
        out.name = 'exp('+out.name+')'
        return out

class KTermFreq(UnaryOpt):
    def __init__(self):
        super(KTermFreq, self).__init__()
        self.name = 'ktermFreq'
        self.type = 1

    def fit(self, value: pd.Series) -> pd.Series:
        shape = value.shape[0]
        tmp = value.value_counts()
        out = value.map(lambda x: tmp[x]/shape) #if x in tmp.index else 0)
        out.name = 'kterm('+value.name+')'
        return out
    
class Ln(UnaryOpt):
    def __init__(self):
        super(Ln, self).__init__()
        self.name = 'ln'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.log(value)
        out.name = 'log('+out.name+')'
        return out


class Negative(UnaryOpt):
    def __init__(self):
        super(Negative, self).__init__()
        self.name = 'negative'
        self.type = 1

    def fit(self, value: pd.Series):
        out = - value
        out.name = '-('+out.name+')'
        return out

class QuanTransform(UnaryOpt):
    def __init__(self):
        super(QuanTransform, self).__init__()
        self.name = 'quanTransform'
        self.type = 1

    def fit(self, value: pd.Series):
        # Quantile Transformer
        scaler = QuantileTransformer()
        out = scaler.fit_transform(value)
        out.name = 'Quan('+value.name+')'
        return out
    
class Radian(UnaryOpt):
    def __init__(self):
        super(Radian, self).__init__()
        self.name = 'radian'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.radians(value)
        out.name = 'radian('+out.name+')'
        return out


class Reciprocal(UnaryOpt):
    def __init__(self):
        super(Reciprocal, self).__init__()
        self.name = 'reciprocal'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.map(lambda x: 1 / x if x != 0 else x)
        out.name = 'reciprocal('+out.name+')'
        return out
    

class Sin(UnaryOpt):
    def __init__(self):
        super(Sin, self).__init__()
        self.name = 'sin'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.sin(value)
        out.name = 'sin('+out.name+')'
        return out


class Sigmoid(UnaryOpt):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = 'sigmoid'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.map(lambda x: 1 / (1 + np.exp(-x)))
        out.name = 'sigmoid('+out.name+')'
        return out


class Square(UnaryOpt):
    def __init__(self):
        super(Square, self).__init__()
        self.name = 'square'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.square(value)
        out.name = 'square('+out.name+')'
        return out
    
    
class Sqrt(UnaryOpt):
    def __init__(self):
        super(Sqrt, self).__init__()
        self.name = 'sqrt'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.sqrt(value)
        out.name = 'sqrt('+out.name+')'
        return out


class Tanh(UnaryOpt):
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = 'tanh'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.tanh(value)
        out.name = 'tanh('+out.name+')'
        return out


class Relu(UnaryOpt):
    def __init__(self):
        super(Relu, self).__init__()
        self.name = 'relu'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.map(lambda x: x * (x > 0))
        out.name = 'relu('+out.name+')'
        return out


## type B: binary
class Div(BinaryOpt):
    def __init__(self):
        self.name = 'div'
        super(Div, self).__init__()
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out = value1 / value2
        out.name = value1.name+'/'+value2.name
        return out

class Minus(BinaryOpt):
    def __init__(self):
        super(Minus, self).__init__()
        self.name = 'minus'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out =  value1 - value2
        out.name = value1.name+'-'+value2.name
        return out

class Add(BinaryOpt):
    def __init__(self):
        super(Add, self).__init__()
        self.name = 'add'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out = value1 + value2
        out.name = value1.name+'+'+value2.name
        return out


class Product(BinaryOpt):
    def __init__(self):
        super(Product, self).__init__()
        self.name = 'product'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out = value1 * value2
        out.name = value1.name+'*'+value2.name
        return out


# type C, multi -> (1,n), return dataframe
class Autoencoder(Transform):
    """!!!ERROR!!!"""
    def __init__(self, dim: int = 32) -> None:
        super(Autoencoder, self).__init__()
        self.name = 'autoencoder'
        self.type = 3

    def fit(self, X_train, X_test):
        # autoencoder
        index_train = X_train.index
        index_test = X_test.index
        dim_in = X_train.shape[1]
        encoding_dim = 32
        input_dat = Input(shape=(dim_in,))
        encoded = Dense(encoding_dim, activation='relu')(input_dat)
        decoded = Dense(dim_in, activation='sigmoid')(encoded)
        autoencoder = Model(input_dat, decoded)
        encoder = Model(input_dat, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='mse')
        autoencoder.fit(X_train, X_train,
                        epochs=50,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=True)
        X_train_ae = pd.DataFrame(encoder.predict(X_train), index=index_train,
                                  columns=['ae_' + str(i) for i in range(encoding_dim)])
        X_test_ae = pd.DataFrame(encoder.predict(X_test), index=index_test,
                                 columns=['ae_' + str(i) for i in range(encoding_dim)])
        X_train = pd.concat([X_train, X_train_ae], axis=1)
        X_test = pd.concat([X_test, X_test_ae], axis=1)
        return X_train, X_test

class Clustering(MultiOpt):
    def __init__(self, k: int = 8, wind_size: int = 5, step: int = 3) -> None:
        """
        clustering with the columns in a window
        """
        super(Clustering, self).__init__()
        self.name = 'clustering'
        self.type = 3
        self.k = k
        self.wind_size = wind_size
        self.step = step

    def fit(self, df: pd.DataFrame) -> pd.Series:
        out = []
        for i in np.arange(0, df.shape[1]-self.wind_size + 1, self.step):
            fname = ''.join([i[0] for i in df.columns])
            cluster = KMeans(n_clusters=self.k).fit(df.iloc[:, i:i+self.wind_size])
            tmp = pd.Series(cluster.predict(df.iloc[:, i:i+self.wind_size]))
            tmp.name = 'Clustering_'+fname
            out.append(tmp)
        return pd.concat(out, axis = 1)

class Diff(MultiOpt):
    def __init__(self):
        """
        diff between the columns
        """
        super(Diff, self).__init__()
        self.name = 'diff'
        self.type = 3

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.diff(axis = 1)
        out.columns = ['diff_' + str(i) for i in out.columns] 
        return out

class KernelApproxRBF(Transform):
    def __init__(self):
        super(KernelApproxRBF, self).__init__()
        self.name = 'kernelapproxrbf'
        self.type = 3

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        rbf_feature = RBFSampler(gamma=1, n_components=df.shape[1])
        res = pd.DataFrame(rbf_feature.fit_transform(df), index=df.index)
        cols = []
        # for i in range(res.shape[1]):
        for i in df.columns:
            cols.append(str(i) + '_rbfFeature')
        res.columns = cols
        return res
    
class LeakyInfoSVR(Transform):
    def __init__(self):
        super(LeakyInfoSVR, self).__init__()
        self.name = 'leakyInfo'
        self.type = 3
        self.condition ='clean data needed, number feature limit'
    
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for i in df.columns:
            cols = df.columns.to_list()
            cols.remove(i)
            df_x = df[cols]
            df_y = df[i]
            svr = SVR()
            svr.fit(df_x, df_y)
            tmp = svr.predict(df_x) -df_y
            tmp.name = str(i) + '_leaky'
            out.append(tmp)
        return pd.concat(out, axis = 1)
    
class Minmaxnorm(MultiOpt):
    def __init__(self) -> None:
        super(Minmaxnorm, self).__init__()
        self.name = 'mmnorm'
        self.type = 3

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = MinMaxScaler()
        out = pd.DataFrame(scaler.fit_transform(df), index=df.index, 
                           columns=[str(i) + '_mmnorm' for i in df.columns])
        return out       

class WinAgg(MultiOpt):
    """regard each item as time series and apply sliding window to it and aggregate"""
    def __init__(self, wind_size: int = 10) -> None:
        super(WinAgg, self).__init__()
        self.name = 'Winagg'
        self.type = 3
        self.wind_size = wind_size

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for ind in np.arange(0, df.shape[1], self.wind_size):
            fname = ''.join([i[0] for i in df.columns])
            tmp = df.iloc[:, ind:ind+self.wind_size].apply(self._agg, axis = 1)
            tmp.columns = [i+'_'+fname for i in ['min', '.25', '.50', '.75', 'max', 'std']]
            out.append(tmp)
        return pd.concat(out, axis =1)
    
    def _agg(self, x):
        return pd.Series([min(x), np.quantile(x, 0.25), np.quantile(x, .5), np.quantile(x, .75), max(x), np.std(x)])
    
class Zscore(MultiOpt):
    def __init__(self):
        super(Zscore, self).__init__()
        self.name = 'zscore'
        self.type = 3
        self.condition = 'data clean needed'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        out = pd.DataFrame(scaler.fit_transform(df), index=df.index, 
                           columns=[str(i) for i in df.columns + '_zscore'])
        return out


class NominalExpansion(Transform):
    """if number of columns smaller then 10 then degree = 3, otherwise degree = 2"""
    def __init__(self):
        super(NominalExpansion, self).__init__()
        self.name = 'nominalExpansion'
        self.type = 2
        self.degree = 2
        self.condition = 'data clean needed'

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        # preprocessing before transformation
        # replace inf with na
        if df.shape[1] <= 10:
            poly = PolynomialFeatures(degree=3, interaction_only=True)
        else:
            poly = PolynomialFeatures(degree=self.degree, interaction_only=True)
        out = pd.DataFrame(poly.fit_transform(df),
                               columns=poly.get_feature_names(input_features=[str(i) for i in df.columns]),
                               index=df.index)
        return out


class IsoMap(MultiOpt):
    def __init__(self):
        super(IsoMap, self).__init__()
        self.name = 'isomap'
        self.type = 3

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        # print(X_test)
        scaler = StandardScaler()
        scaler.fit(df)
        df_z = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
        # print(X_test_z)
        # isomap
        nc = 16
        if min(df.shape) < nc:
            nc = min(df.shape)
        embedding = Isomap(n_components=nc).fit(df_z)
        df_t = pd.DataFrame(embedding.transform(df_z), index=df_z.index,
                                 columns=['isomap_' + str(i) for i in range(nc)])
        return df_t

# type D, need help of y
### fail noch the fit method for type D and type E
### type E: feature engineering with model

class DecisionTreeClassifierTransform(Transform):
    def __init__(self):
        super(DecisionTreeClassifierTransform, self).__init__()
        self.name = 'decisionTreeClassifierTransform'
        self.type = 4
 
    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cols = [3,4,6,7,10,11,13,14]
        ### extract nodes sample
        reg = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
        # extract information from tree
        dec_paths = reg.decision_path(X_train).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_train_samples = pd.DataFrame(tmp / tmp[0][0], index=X_train.index,
                                       columns=['dt_n_cla_' + str(i) for i in range(tmp.shape[1])])
        series_cluster_train = X_train_samples.iloc[:, cols].apply(lambda x: np.argmax(x), axis = 1)
        series_cluster_train.name = 'dt_cla_cluster'
        dec_paths = reg.decision_path(X_test).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_test_samples = pd.DataFrame(tmp / tmp[0][0], index=X_test.index,
                                      columns=['dt_n_cla_' + str(i) for i in range(tmp.shape[1])])
        series_cluster_test = X_test_samples.iloc[:, cols].apply(lambda x: np.argmax(x), axis = 1)
        series_cluster_test.name = 'dt_cla_cluster'
        ### train dt model and predict
        reg = DecisionTreeClassifier().fit(X_train, y_train)
        # predict
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_cla_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['dt_cla_pred']).unique()) == 1:
            out_train = pd.concat([series_cluster_train, X_train_pred], axis = 1)
            out_test = pd.concat([series_cluster_test, X_test_pred], axis = 1)
            return out_train, out_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = DecisionTreeClassifier().fit(X_train, y_train == X_train_pred['dt_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_cla_pred_diff'])
        #X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        #X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        out_train = pd.concat([series_cluster_train, X_train_pred, X_train_pred_diff], axis =1)
        out_test = pd.concat([series_cluster_test, X_test_pred, X_test_pred_diff], axis = 1)
        ### extract information from tree
        return out_train, out_test


class DecisionTreeRegressorTransform(Transform):
    def __init__(self):
        super(DecisionTreeRegressorTransform, self).__init__()
        self.name = 'decisionTreeRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        cols = [3,4,6,7,10,11,13,14]
        ### extract nodes sample
        reg = DecisionTreeRegressor(max_depth=4).fit(X_train, y_train)
        # extract information from tree
        dec_paths = reg.decision_path(X_train).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_train_samples = pd.DataFrame(tmp / tmp[0][0], index=X_train.index,
                                       columns=['dt_n_reg_' + str(i) for i in range(tmp.shape[1])])
        series_cluster_train = X_train_samples.iloc[:, cols].apply(lambda x: np.argmax(x), axis = 1)
        series_cluster_train.name = 'dt_cla_cluster'
        dec_paths = reg.decision_path(X_test).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_test_samples = pd.DataFrame(tmp / tmp[0][0], index=X_test.index,
                                      columns=['dt_n_reg_' + str(i) for i in range(tmp.shape[1])])
        series_cluster_test = X_test_samples.iloc[:, cols].apply(lambda x: np.argmax(x), axis = 1)
        series_cluster_test.name = 'dt_cla_cluster'
        ### train dt model and predict
        reg = DecisionTreeClassifier().fit(X_train, y_train)
        # predict
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_reg_pred'])
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = DecisionTreeClassifier().fit(X_train, y_train - X_train_pred['dt_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_reg_pred_diff'])
        ### extract information from tree
        out_train = pd.concat([series_cluster_train, X_train_pred, X_train_pred_diff], axis =1)
        out_test = pd.concat([series_cluster_test, X_test_pred, X_test_pred_diff], axis = 1)
        return X_train, X_test


class LinearRegressorTransform(Transform):
    def __init__(self):
        super(LinearRegressorTransform, self).__init__()
        self.name = 'linearRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = LinearRegression().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['linear_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['linear_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = LinearRegression().fit(X_train, y_train - X_train_pred['linear_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['linear_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['linear_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class MLPClassifierTransform(Transform):
    def __init__(self):
        super(MLPClassifierTransform, self).__init__()
        self.name = 'mlpClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        train_dataset = Data_rb_cla(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        model = MLP(X_train.shape[1], hidden_size=32, output_size=len(y_train.unique()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        model = self._train_model(model, epochs=100, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
        X_train_pred = pd.DataFrame(model.layers(torch.from_numpy(X_train.to_numpy()).float()).data.numpy(),
                                    index=X_train.index,
                                    columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        X_test_pred = pd.DataFrame(model.layers(torch.from_numpy(X_test.to_numpy()).float()).data.numpy(),
                                   index=X_test.index,
                                   columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        return X_train_pred, X_test_pred

    def _train_model(self, model, epochs, train_loader, optimizer, loss_fn):
        for epoch in range(epochs):
            model.train()
            losses = []
            for i, (Xs, ys) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(Xs)
                loss = loss_fn(outputs, ys)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # print(np.mean(losses))
        return model


class MLPRegressorTransform(Transform):
    def __init__(self):
        """
        the different between MLPReg and MLPCla is the loss function and y value
        """
        super(MLPRegressorTransform, self).__init__()
        self.name = 'mlpRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        train_dataset = Data_rb_reg(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        model = MLP(X_train.shape[1], hidden_size=32, output_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.L1Loss()
        model = self._train_model(model, epochs=100, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
        X_train_pred = pd.DataFrame(model.layers(torch.from_numpy(X_train.to_numpy()).float()).data.numpy(),
                                    index=X_train.index,
                                    columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        X_test_pred = pd.DataFrame(model.layers(torch.from_numpy(X_test.to_numpy()).float()).data.numpy(),
                                   index=X_test.index,
                                   columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        return X_train_pred, X_test_pred

    def _train_model(self, model, epochs, train_loader, optimizer, loss_fn):
        for epoch in range(epochs):
            model.train()
            losses = []
            for i, (Xs, ys) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(Xs)
                loss = loss_fn(outputs, ys)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            #print(np.mean(losses))
        return model


class MLP(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.ln = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        '''
        inp shape of torch tensor
        '''
        out = self.ln(self.layers(inp))
        return out


class Data_rb_cla(Dataset):
    def __init__(self, Xs, ys):
        # input is type of pandas
        self.Xs = torch.from_numpy(Xs.to_numpy()).float()
        self.ys = torch.from_numpy(ys.to_numpy()).long()

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]


class Data_rb_reg(Dataset):
    def __init__(self, Xs, ys):
        # input is type of pandas
        self.Xs = torch.from_numpy(Xs.to_numpy()).float()
        self.ys = torch.from_numpy(ys.to_numpy()).float()

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]


class NearestNeighborsClassifierTransform(Transform):
    def __init__(self):
        super(NearestNeighborsClassifierTransform, self).__init__()
        self.name = 'nearestNeighborsClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        neigh = NearestNeighbors()
        neigh.fit(X_train)
        neigh.kneighbors(n_neighbors=30)
        neigh.fit(X_train)
        indx = pd.DataFrame(neigh.kneighbors(X_train)[1], index=X_train.index)
        nnx = indx.apply(lambda x: pd.Series([(y_train.iloc[x] == i).sum() for i in range(len(y_train.unique()))]),
                         axis=1)
        nnx.columns = ['nn_' + str(i) for i in nnx.columns]
        indy = pd.DataFrame(neigh.kneighbors(X_test)[1], index=X_test.index)
        nny = indy.apply(lambda x: pd.Series([(y_train.iloc[x] == i).sum() for i in range(len(y_train.unique()))]),
                         axis=1)
        nny.columns = ['nn_' + str(i) for i in nny.columns]
        #X_train = pd.concat([X_train, nnx], axis=1)
        #X_test = pd.concat([X_test, nny], axis=1)
        return nnx, nny


class NearestNeighborsRegressorTransform(Transform):
    def __init__(self):
        super(NearestNeighborsRegressorTransform, self).__init__()
        self.name = 'nearestNeighborsRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        neigh = NearestNeighbors()
        neigh.fit(X_train)
        neigh.kneighbors(n_neighbors=5)
        neigh.fit(X_train)
        indx = pd.DataFrame(neigh.kneighbors(X_train)[1], index=X_train.index)
        out_train = pd.DataFrame(indx.apply(lambda x: np.mean([y_train.iloc[i] for i in x]), axis=1), columns = ['nn_target'])
        indy = pd.DataFrame(neigh.kneighbors(X_test)[1], index=X_test.index)
        out_test = pd.DataFrame(indy.apply(lambda x: np.mean([y_train.iloc[i] for i in x]), axis=1), columns = ['nn_target'])
        return out_train, out_test


class SVRTransform(Transform):
    def __init__(self):
        super(SVRTransform, self).__init__()
        self.name = 'svrTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = SVR().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svr_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svr_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = SVR().fit(X_train, y_train - X_train_pred['svr_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svr_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svr_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class SVCTransform(Transform):
    def __init__(self):
        super(SVCTransform, self).__init__()
        self.name = 'svcTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = SVC().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svc_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svc_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['svc_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = SVC().fit(X_train, y_train == X_train_pred['svc_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svc_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svc_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class GauDotWhiteRegressorTransform(Transform):
    def __init__(self):
        super(GauDotWhiteRegressorTransform, self).__init__()
        self.name = 'gauDotWhiteRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaudotwhite_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_reg_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train - X_train_pred['gaudotwhite_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['gaudotwhite_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_reg_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class GauDotClassifierTransform(Transform):
    def __init__(self):
        super(GauDotClassifierTransform, self).__init__()
        self.name = 'gauDotClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaudotwhite_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_cla_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['gaudotwhite_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train == X_train_pred['gaudotwhite_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['gaudotwhite_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_cla_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class GauExpRegressorTransform(Transform):
    def __init__(self):
        super(GauExpRegressorTransform, self).__init__()
        self.name = 'gauExpRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = Exponentiation(RationalQuadratic(), exponent=2)
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_reg_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train - X_train_pred['gauexp_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_reg_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauExpClassifierTransform(Transform):
    def __init__(self):
        super(GauExpClassifierTransform, self).__init__()
        self.name = 'gauExpClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = Exponentiation(RationalQuadratic(), exponent=2)
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_cla_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['gauexp_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train == X_train_pred['gauexp_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_cla_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class GauRBFRegressorTransform(Transform):
    def __init__(self):
        super(GauRBFRegressorTransform, self).__init__()
        self.name = 'gauRBFRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = 1.0 * RBF(1.0)
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_reg_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train - X_train_pred['gaurbf_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_reg_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class GauRBFClassifierTransform(Transform):
    def __init__(self):
        super(GauRBFClassifierTransform, self).__init__()
        self.name = 'gauRBFClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = 1.0 * RBF(1.0)
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_cla_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['gaurbf_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train == X_train_pred['gaurbf_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_cla_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class RandomForestClassifierTransform(Transform):
    def __init__(self):
        super(RandomForestClassifierTransform, self).__init__()
        self.name = 'randomForestClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = RandomForestClassifier().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfc_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfc_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['rfc_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = RandomForestClassifier().fit(X_train, y_train == X_train_pred['rfc_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfc_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfc_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class RandomForestRegressorTransform(Transform):
    def __init__(self):
        super(RandomForestRegressorTransform, self).__init__()
        self.name = 'randomForestRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = RandomForestRegressor().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfr_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfr_pred'])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = RandomForestRegressor().fit(X_train, y_train - X_train_pred['rfr_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfr_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfr_pred_diff'])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class XGBClassifierTransform(Transform):
    def __init__(self):
        super(XGBClassifierTransform, self).__init__()
        self.name = 'xgbClassifierTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = xgb.XGBClassifier().fit(X_train, y_train)
        ppname = str(time.time())[-3:]
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['xgb_cla_pred' + ppname])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_cla_pred' + ppname])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['xgb_cla_pred' + ppname]).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = xgb.XGBClassifier().fit(X_train, y_train == X_train_pred['xgb_cla_pred' + ppname])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['xgb_cla_pred_diff' + ppname])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_cla_pred_diff' + ppname])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test


class XGBRegressorTransform(Transform):
    def __init__(self):
        super(XGBRegressorTransform, self).__init__()
        self.name = 'xgbRegressorTransform'
        self.type = 4

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = xgb.XGBRegressor().fit(X_train, y_train)
        ppname = str(time.time())[-3:]
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['xgb_reg_pred' + ppname])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_reg_pred' + ppname])
        #X_train = pd.concat([X_train, X_train_pred], axis=1)
        #X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = xgb.XGBRegressor().fit(X_train, y_train - X_train_pred['xgb_reg_pred' + ppname])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['xgb_reg_pred_diff' + ppname])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_reg_pred_diff' + ppname])
        out_train = pd.concat([X_train_pred, X_train_pred_diff], axis=1)
        out_test = pd.concat([X_test_pred, X_test_pred_diff], axis=1)
        return out_train, out_test