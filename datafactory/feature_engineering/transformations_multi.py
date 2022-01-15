from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
    
class MultiOpt(metaclass=ABCMeta):
    
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

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
            tmp.name = 'Clustering_' + fname
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
    
class IsoMap(MultiOpt):
    
    def __init__(self):
        super(IsoMap, self).__init__()
        self.name = 'isomap'
        self.type = 3

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        scaler.fit(df)
        df_z = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
        # isomap
        nc = 16
        if min(df.shape) < nc:
            nc = min(df.shape)
        embedding = Isomap(n_components=nc).fit(df_z)
        df_t = pd.DataFrame(embedding.transform(df_z), index=df_z.index,
                                 columns=['isomap_' + str(i) for i in range(nc)])
        return df_t