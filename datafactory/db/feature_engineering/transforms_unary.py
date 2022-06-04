'''
Copyright (c) Smart Data Solution Center Baden-Württemberg 2021,
All rights reserved.
'''

from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

class UnaryOpt(metaclass=ABCMeta):
    
    @abstractmethod
    def fit(self, value: pd.Series) -> pd.Series:
        pass

class Abs(UnaryOpt):
    
    def __init__(self):
        super(Abs, self).__init__()
        self.name = 'abs'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.abs()
        out.name = 'abs(' + out.name + ')'
        return out

class Add(UnaryOpt):
    
    def __init__(self):
        super(Add, self).__init__()
        self.name = 'add'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value + np.e
        out.name = out.name + '+e'
        return out

class Cos(UnaryOpt):
    
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.cos(value)
        out.name = 'cos(' + out.name + ')'
        return out

class Degree(UnaryOpt):
    
    def __init__(self):
        
        super(Degree, self).__init__()
        self.name = 'degree'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.degrees(value)
        out.name = 'degree(' + out.name + ')'
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
        out.name = 'exp(' + out.name + ')'
        return out

class KTermFreq(UnaryOpt):
    
    def __init__(self):
        super(KTermFreq, self).__init__()
        self.name = 'ktermFreq'
        self.type = 1

    def fit(self, value: pd.Series) -> pd.Series:
        shape = value.shape[0]
        tmp = value.value_counts()
        out = value.map(lambda x: tmp[x]/shape) #if x in tmp.index else 0
        out.name = 'kterm(' + value.name + ')'
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
        out.name = '-(' + out.name + ')'
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
        out.name = 'Quan(' + value.name+ ')'
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
        out.name = 'reciprocal(' + out.name + ')'
        return out
    

class Sin(UnaryOpt):
    
    def __init__(self):
        super(Sin, self).__init__()
        self.name = 'sin'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.sin(value)
        out.name = 'sin(' + out.name + ')'
        return out


class Sigmoid(UnaryOpt):
    
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = 'sigmoid'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.map(lambda x: 1 / (1 + np.exp(-x)))
        out.name = 'sigmoid(' + out.name + ')'
        return out


class Square(UnaryOpt):
    
    def __init__(self):
        super(Square, self).__init__()
        self.name = 'square'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.square(value)
        out.name = 'square(' + out.name + ')'
        return out
    
    
class Sqrt(UnaryOpt):
    
    def __init__(self):
        super(Sqrt, self).__init__()
        self.name = 'sqrt'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.sqrt(value)
        out.name = 'sqrt(' + out.name + ')'
        return out


class Tanh(UnaryOpt):
    
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = 'tanh'
        self.type = 1

    def fit(self, value: pd.Series):
        out = np.tanh(value)
        out.name = 'tanh(' + out.name + ')'
        return out


class Relu(UnaryOpt):
    
    def __init__(self):
        super(Relu, self).__init__()
        self.name = 'relu'
        self.type = 1

    def fit(self, value: pd.Series):
        out = value.map(lambda x: x * (x > 0))
        out.name = 'relu(' + out.name + ')'
        return out