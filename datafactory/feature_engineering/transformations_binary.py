from abc import ABCMeta, abstractmethod

import pandas as pd

class BinaryOpt(metaclass=ABCMeta):
    
    @abstractmethod
    def fit(self, value1: pd.Series, value2: pd.Series) -> pd.Series:
        pass

class Add(BinaryOpt):
    
    def __init__(self):
        super(Add, self).__init__()
        self.name = 'add'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out = value1 + value2
        out.name = value1.name + '+' + value2.name
        return out

class Minus(BinaryOpt):
    
    def __init__(self):
        super(Minus, self).__init__()
        self.name = 'minus'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out =  value1 - value2
        out.name = value1.name + '-' + value2.name
        return out
    
class Product(BinaryOpt):
    
    def __init__(self):
        super(Product, self).__init__()
        self.name = 'product'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out = value1 * value2
        out.name = value1.name + '*' + value2.name
        return out#
    
class Div(BinaryOpt):
    
    def __init__(self):
        super(Div, self).__init__()
        self.name = 'div'
        self.type = 2

    def fit(self, value1: pd.Series, value2: pd.Series):
        out = value1 / value2
        out.name = value1.name + '/' + value2.name
        return out