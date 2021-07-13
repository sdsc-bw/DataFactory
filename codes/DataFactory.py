import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from abc import ABCMeta, abstractmethod
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import cast, Any, Dict, List, Tuple, Optional, Union
from transforms import UnaryOpt, BinaryOpt, MultiOpt
import transforms as tfd
import logging

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
        
        
    def pipeline(self, fn: str) -> Tuple[list, list]:
        """load data + compute baseline + generate meta feature + test transformation + wrap result"""
        self.logger.info('='*60)
        self.logger.info(f'Extract train data from: {fn}')
        self.logger.info('='*60)
        self.logger.info('+ Start to load data, preprocessing and calculate baseline')
        dat, target, baseline = self.load_data(fn)
        self.logger.info(f'- End with load data: shape {dat.shape}, baseline {baseline}')
        cols = dat.columns
        
    
    def apply_binary_transformations_for_series(self, value1: pd.Series, value2: pd.Series) -> pd.DataFrame:
        values = []
        self.logger.info(f'+ Start to apply binary transformtions to series {value1.name} and series {value2.name}')
        for key in self.opts_binary.keys():
            self.logger.info(f'    applying transformation: {self.opts_binary[key].name}')
            tmp_value = self.opts_binary[key].fit(value1, value2)
            values.append(tmp_value)
        self.logger.info(f'- Finish transformation')
        return pd.concat(values, axis = 1)
            
    def apply_multiple_transformations_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = []
        self.logger.info(f'+ Start to apply multi transformtions to dataframe')
        for key in self.opts_multi.keys():
            self.logger.info(f'    applying transformation: {self.opts_multi[key].name}')
            tmp_df = self.opts_multi[key].fit(df)
            dfs.append(tmp_df)
        self.logger.info(f'- Finish transformation')
        return pd.concat(dfs, axis = 1)
    
    def apply_supervised_transformations_for_dataframe(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                                       y_train: pd.Series, y_test: pd.Series, art = 'C') -> Tuple[pd.DataFrame, pd.DataFrame]:
        dfs_train, dfs_test = [], []
        self.logger.info(f'+ Start to apply supervised transformtions to dataframe')
        opts = self.opts_cla_supervised if art == 'C' else self.opts_reg_supervised
        for key in opts.keys():
            self.logger.info(f'    applying transformation: {opts[key].name}')
            tmp_train, tmp_test = opts[key].fit(X_train, X_test, y_train, y_test)
            dfs_train.append(tmp_train)
            dfs_test.append(tmp_test)
        self.logger.info(f'- Finish transformation')
        return pd.concat(dfs_train, axis = 1), pd.concat(dfs_test, axis = 1)
    
    def apply_unary_transformations_to_series(self, value: pd.Series) -> pd.DataFrame:
        values = []
        self.logger.info(f'+ Start to apply unary transformtions to series: {value.name}')
        for key in self.opts_unary.keys():
            self.logger.info(f'    applying transformation: {self.opts_unary[key].name}')
            tmp_value = self.opts_unary[key].fit(value)
            values.append(tmp_value)
        self.logger.info(f'- Finish transformation')
        return pd.concat(values, axis = 1)
        
    def clean_dat(self, dat: pd.DataFrame) -> pd.DataFrame:
        """deal with nan and inf"""
        if dat.empty:
            return dat
        self.logger.info('+ Start to clean the given dataframe')
        self.logger.info('    number of inf and nan are for dataset: (%d, %d)' % ((dat == np.inf).sum().sum(), dat.isna().sum().sum()))
        self.logger.info('    set type to float32 at first && deal with inf.')
        dat = dat.astype(np.float32)
        dat = dat.replace([np.inf, -np.inf], np.nan)
        self.logger.info('    remove columns with half of nan')
        dat = dat.dropna(axis=1, thresh=dat.shape[0] * .5)
        self.logger.debug('    remove costant columns')
        dat = dat.loc[:, (dat != dat.iloc[0]).any()]

        if dat.isna().sum().sum() > 0:
            logger.debug('Start to fill the columns with nan')
            # imp = IterativeImputer(max_iter=10, random_state=0)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            # dat = dat.fillna(dat.mean())
            tmp = imp.fit_transform(dat)
            if tmp.shape[1] != dat.shape[1]:
                tmp = dat.fillna(0)
            dat = pd.DataFrame(tmp, columns=dat.columns, index=dat.index)
        #logger.info('Remove rows with any nan in the end')
        #dat = dat.dropna(axis=0, how='any')
        self.logger.info('- Finish with Data cleaning, number of inf and nan are for dataset: (%d, %d)' 
                     % ((dat == np.inf).sum().sum(), dat.isna().sum().sum()))
        dat = dat.reset_index(drop=True)
        return dat
        
    def evaluate(self, dat: pd.DataFrame, target: pd.Series, cv: int = 5, art = 'C') -> Tuple[float, float]:
        """evaluate a data set with random forest"""
        scores = []
        for i in range(cv):
            X_train, X_test, y_train, y_test = train_test_split(dat, target, random_state = i)
            if art == 'C':
                model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                score = f1_score(y_test, predict, average='weighted')
            elif art == 'R':
                model = RandomForestRegressor(random_state = self.rf_seed)
                model.fit(X_train, y_train)
                predict = model.predict(X_test)
                score = 1 - self._relative_absolute_error(predict, y_test)
            else:
                self.logger.error('The type of the task is unknown')
            scores.append(score)
        return np.mean(scores), np.std(scores)
    
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
    
    def plot_linear_for_columns_in_df(self, df, step, cols, save_path = None, id = None):
        """
        drop linear plot for columns A and B in dataframe df.
        parameter:
        =============
        df, type of DataFrame
            - target data frame
        step, type of int
            - plot the value every ||step|| items
        cols, multi parameter
            - each steht for a columns that needs to plot in the figure
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
        
    def plot_density_for_each_column_in_df(self, df, save_path = None, id = None):
        """
        drop linear plot for columns A and B in dataframe df.
        parameter:
        =============
        df, type of DataFrame
            - target data frame
        step, type of int
            - plot the value every ||step|| items
        cols, multi parameter
            - each steht for a columns that needs to plot in the figure
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
    
    def _relative_absolute_error(self, pred, y):
        dis = abs((pred-y)).sum()
        dis2 = abs((y.mean() - y)).sum()
        #print(dis, dis2)
        if dis2 == 0 :
            return 1
        return dis/dis2