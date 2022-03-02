# data process packages
import copy
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# logging setting
import logging

# thread and timer
from threading import Timer

# plot packages
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#from wordcloud import WordCloud

# other
import argparse
import pytorchcv
import os
import sys
sys.path.append('../../DataFactory-develop 2/')

from datafactory.ts.preprocessing.loading import * # methods for dataset loading
from datafactory.ts.preprocessing.encoding import * # methods for encoding
from datafactory.ts.preprocessing.outlier_detecting import outlier_detection_feature, outlier_detection_dataframe # methods for outlier detection
from datafactory.ts.preprocessing.cleaning import * # methods for data cleaning
from datafactory.ts.preprocessing.sampling import * # mehtods for sampling
from datafactory.ts.preprocessing.validating import * # methods for data checking
from datafactory.ts.plotting.model_plotting import * # plot method

# model package
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor 
from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# plot
from sklearn import tree
import graphviz
import numpy as np
import matplotlib.pyplot as plt

from dtreeviz.trees import dtreeviz # remember to load the package
from tqdm import tqdm
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")







parser = argparse.ArgumentParser("relational database process")
parser.add_argument('--datapath', type=str, required=True, default='', help='the path of the data')
parser.add_argument('--targetname', type=str, help='the name of the target feature')
parser.add_argument('--art', type=str,  default = 'C', help='the art of the task, either C or R')
parser.add_argument('--sep', type=str, default=';', help='the seperator of the target file')
parser.add_argument('--index_col', type=int, default=0, help='the location of the index column')
parser.add_argument('--outputpath', type=str, default='./', help='the path to save the output file')
args = parser.parse_args()








def main():
    # get argument from comment line    
    datapath = args.datapath
    targetname = args.targetname
    art = args.art
    sep = args.sep
    index_col = args.index_col
    outputpath = args.outputpath
    
    # check dir existent
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    
    if not os.path.isdir(outputpath+'/plots/'):
        os.mkdir(outputpath+'/plots/')
    
    if not os.path.isdir(outputpath+'/plots/scatter_att/'):
        os.mkdir(outputpath+'/plots/scatter_att/')
    
    if not os.path.isdir(outputpath+'/plots/scatter_y/'):
        os.mkdir(outputpath+'/plots/scatter_y/')
    
    if not os.path.isdir(outputpath+'/plots/class_based_distribution/'):
        os.mkdir(outputpath+'/plots/class_based_distribution/')
    
    # open new file to save the output:
    infofile = open(outputpath + "/data_report.txt", "w")
    
    # load data
    df = pd.read_csv(datapath, sep=sep, index_col=index_col)
    
    #### check data
    # check data shape 
    if df.shape[1] < 2:
        print(f'The number of features found in the dataset is {df.shape[2]}, it may due to the wrong setting of seperator, please rerun the programm and set the seperator with parameter --sep')
    
    df.describe().to_csv(outputpath+'statistic.csv')
    
    # check data
    dat_numeric, dat_categoric, dat_y, le_name_mapping, flag_balance, flag_wrong_target = check_data_and_distribute(df, art = 'C', file = infofile, logger = logger)
    
    inp = 'yes'
    if flag_wrong_target:
        inp = input("there are more then 10 classed in the given dataset that contain less than 10 items. We suggestion that the given target may be wrong, please check the data: return 'yes' to continue the program and all the other to stop the program")
        
    if inp == 'yes':
        print('Continue with the given target')
    else:
        sys.exit()
    
    # na/inf cleaning
    dat_numeric = clean_data(dat_numeric)
    
    # label encoding
    dat_numeric2 = categorical_feature_encoding(dat_categoric)
    
    # combine dat_categoric and dat_numeric
    dat = pd.concat([dat_numeric, dat_numeric2], axis = 1)
    
    # outliear detection
    outlier = outlier_detection_dataframe(dat)
    outlier.to_csv(outputpath+'outlier.csv')
    
    #### plots
    # show feature importance
    fig, feature_importances = plot_feature_importance_of_random_forest(dat, dat_y)#, strategy = 'permutation')
    fig.write_image(outputpath+'/plots/feature_importance.webp')
    
    # show heatmap
    fig = px.imshow(dat.corr())
    fig.update_xaxes(side="bottom")
    fig.write_image(outputpath+'/plots/correlation_heatmap.webp')
    
    # subset when too many features
    if dat.shape[1]>10:
      print("More than 10 features found in the given dataset, Only consider the 10 most important features")
      tmp_dat = dat[feature_importances.index[:10]]

    else:
      tmp_dat = copy.deepcopy(dat)
    
    # show scatter plot
    leng = tmp_dat.shape[1]
    for i in range(leng):
        for j in range(leng):
            fig = px.scatter(tmp_dat, x=tmp_dat.columns[i], y=tmp_dat.columns[j],marginal_x="histogram", marginal_y="histogram")
            fig.write_image(outputpath+'/plots/scatter_att/' + tmp_dat.columns[i].replace('/', '')+'-'+tmp_dat.columns[j].replace('/', '')+'.webp')
            
    # show relationship between attributes and y
    for col in dat.columns:
        fig = px.scatter(dat, x=col, y=dat_y, marginal_x="histogram", marginal_y="histogram")
        fig.write_image(outputpath+'/plots/scatter_y/' + col.replace('/', '')+'-target.webp')
        
    # box or violine plot, after normalization
    tmp = (tmp_dat-tmp_dat.mean())/tmp_dat.std()
    tmp2 = []

    for i in tmp.columns:
      tmp3 = pd.DataFrame(columns = ['value', 'fname'])
      tmp3['value'] = tmp[i].values
      tmp3['fname'] = i#[i for j in range(tmp3.shape[0])]

      tmp2.append(tmp3)

    tmp = pd.concat(tmp2, axis = 0)

    fig = go.Figure()
    for i in tmp_dat.columns:
        fig.add_trace(go.Violin(y=tmp['value'][tmp['fname'] == i], x= tmp['fname'][tmp['fname'] == i],
              name = i,
              box_visible=True, 
              #line_color='black',
              meanline_visible=True, #fillcolor='lightseagreen', 
              #opacity=0.6
              ))

    fig.write_image(outputpath+'/plots/violin_features.webp')
    
    inv_le_name_mapping = {}
    for i, j in le_name_mapping.items():
        inv_le_name_mapping[j] = i
    
    # C distribution
    if art == 'C':
        for col in tmp_dat.columns:
            fig = go.Figure()

            for i in inv_le_name_mapping.keys():
                if col == tmp_dat.columns[0]:
                    flag = True

                fig.add_trace(go.Violin(y=tmp_dat[col][dat_y == i], x= pd.Series(dat_y[dat_y == i]).map(lambda x: inv_le_name_mapping[x]),
                    name = inv_le_name_mapping[i],
                    box_visible=True, 
                    points='all',
                    #line_color='black',
                    meanline_visible=True, #fillcolor='lightseagreen', 
                    #legendgroup='group',
                    showlegend=flag
                    ))

            fig.write_image(outputpath+'/plots/class_based_distribution/' + col.replace('/', '') + '.webp')

    # build model
    if art == 'C':
        out, fig = basic_model_comparison_classification(dat, dat_y)
    elif art == 'R':
        out, fig = basic_model_comparison_regression(dat, dat_y)
    else:
        print(f'Unrecognized art {art}, use regression instead')
        out, fig = basic_model_comparison_regression(dat, dat_y)
    
    out.to_csv(outputpath + '/performance_comparison.csv')
    fig.write_image(outputpath+'/plots/performance_comparison.webp')
    
    #visulize decition tree
    if art == 'C':
        fig1, fig2 = plot_decision_tree_classification(dat, dat_y)
    elif art == 'R':
        fig1, fig2 = plot_decision_tree_regression(dat, dat_y)
    else:
        print(f'Unrecognized art {art}, use regression instead')
        fig1, fig2 = plot_decision_tree_regression(dat, dat_y)
    fig1.save(outputpath+'/plots/decisiontree_explaination1.pdf')
    fig2.save(outputpath+'/plots/decisiontree_explaination2.svg')

if __name__ == '__main__':
    main()
    
