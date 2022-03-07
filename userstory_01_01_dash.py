# dash
import dash

from dash import dcc, html
from dash.dependencies import Input, Output

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

from datafactory.ts.preprocessing.encoding import * # methods for encoding
from datafactory.ts.preprocessing.outlier_detecting import outlier_detection_feature, outlier_detection_dataframe # methods for outlier detection
from datafactory.ts.preprocessing.cleaning import * # methods for data cleaning
from datafactory.ts.preprocessing.sampling import * # mehtods for sampling
from datafactory.ts.preprocessing.validating import * # methods for data checking
from datafactory.ts.plotting.model_plotting import * # plot method


# plot tree
from sklearn import tree
import graphviz
import numpy as np
import matplotlib.pyplot as plt

from dtreeviz.trees import dtreeviz # remember to load the package
from tqdm import tqdm
from matplotlib.colors import ListedColormap

import warnings
warnings.filterwarnings("ignore")

## setup parser
parser = argparse.ArgumentParser("relational database process")
parser.add_argument('--datapath', type=str, required=True, default='', help='the path of the data')
parser.add_argument('--targetname', type=str, help='the name of the target feature')
parser.add_argument('--art', type=str,  default = 'C', help='the art of the task, either C or R')
parser.add_argument('--sep', type=str, default=';', help='the seperator of the target file')
parser.add_argument('--index_col', type=int, default=0, help='the location of the index column')
parser.add_argument('--outputpath', type=str, default='./', help='the path to save the output file')
args = parser.parse_args()

## Setup dash
app = dash.Dash(__name__)
#app.layout = html.H1('This is my first DASH Application')

# get argument from comment line    
datapath = args.datapath
targetname = args.targetname
art = args.art
sep = args.sep
index_col = args.index_col
outputpath = args.outputpath


#### ATTENTION-IDEA: maybe, instead of the waterflow layout, use dash Tabs()+Tab() to create a better display

def main():
    
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
    
    # load data
    df = pd.read_csv(datapath, sep=sep, index_col=index_col)
    
    #### check data
    # check data shape 
    if df.shape[1] < 2:
        print(f'The number of features found in the dataset is {df.shape[2]}, it may due to the wrong setting of seperator, please rerun the programm and set the seperator with parameter --sep')
    
    # statistic information
    dat_describe = df.describe()
    dat_describe.to_csv(outputpath+'statistic.csv')
    dat_describe = pd.read_csv(outputpath+'statistic.csv')
    
    # check data, 
    infofile = open(outputpath + "/data_report.txt", "w")
    dat_numeric, dat_categoric, dat_y, dat_number_na, le_name_mapping, flag_balance, flag_wrong_target = check_data_and_distribute(df, art = 'C', file = infofile, logger = logger)
    infofile.close()
    dat_number_na.to_csv(outputpath+'/number_na.csv')
    
    dat_number_na = pd.read_csv(outputpath+'/number_na.csv')
    dat_number_na.columns = ['feature', 'number of na']
    dat_number_na = dat_number_na[dat_number_na['number of na'].map(lambda x: float(x.split('/')[0])/float(x.split('/')[1])) > 0]
    
    inp = 'yes'
    if flag_wrong_target:
        inp = input("there are more then 10 classed in the given dataset that contain less than 10 items. We suggestion that the given target may be wrong, please check the data: return 'yes' to continue the program and all the other to stop the program")
        
    if inp == 'yes':
        print('Continue with the given target')
    else:
        sys.exit()
    
    # na/inf cleaning
    infofile = open(outputpath + "/data_process.txt", "w")
    dat_numeric = clean_data(dat_numeric, file = infofile)
    
    ###TODO!!!!: add explanation for k-term feature engineering
    # label encoding
    dat_numeric2 = categorical_feature_encoding(dat_categoric, file = infofile)
    
    # combine dat_categoric and dat_numeric
    dat = pd.concat([dat_numeric, dat_numeric2], axis = 1)
    print(f'Shape of dataframe after processing is ***{dat.shape}***')
    
    infofile.close()
    
    ##TODO!!!!: missing fig to show the outlier, maybe box plot with single feature detection
    # outliear detection
    outlier = outlier_detection_dataframe(dat) # maybe should use dat_numeric as input instead of the whole dat
    outlier.to_csv(outputpath+'outlier.csv')
    
    # show feature importance
    fig_feature_importances, feature_importances = plot_feature_importance_of_random_forest(dat, dat_y)#, strategy = 'permutation')
    fig_feature_importances.write_image(outputpath+'/plots/feature_importance.webp')
    feature_importances.to_csv(outputpath+'/feature_importances.csv')
    feature_importances = pd.read_csv(outputpath+'/feature_importances.csv')
    feature_importances.columns = ['feature', 'importance']
    
    # show heatmap
    tmp_heatmap = dat.corr()
    fig_heatmap = px.imshow(tmp_heatmap)
    fig_heatmap.update_xaxes(side="bottom")
    fig_heatmap.write_image(outputpath+'/plots/correlation_heatmap.webp')
    
    ###TODO!!!!:  no idea how to extract the importance information automatically form the scatter plot.
    # subset when too many features
    tmp_dat = None
    if dat.shape[1]>10:
      print("More than 10 features found in the given dataset, Only consider the 10 most important features")
      tmp_dat = dat[feature_importances['feature'][:10]]

    else:
      tmp_dat = copy.deepcopy(dat)
    
    ###TODO!!!!: show the importance scatter(high correlation features) before the selection part
    # show scatter plot, save to results.
    leng = tmp_dat.shape[1]
    ###TODO!!!!: uncomment when using 
    #for i in range(leng):
    #    for j in range(leng):
    #        fig = px.scatter(tmp_dat, x=tmp_dat.columns[i], y=tmp_dat.columns[j],marginal_x="histogram", marginal_y="histogram")
    #        fig.write_image(outputpath+'/plots/scatter_att/' + tmp_dat.columns[i].replace('/', '')+'-'+tmp_dat.columns[j].replace('/', '')+'.webp')
    
    # show relationship between attributes and y
    for col in dat.columns:
        fig = px.scatter(dat, x=col, y=dat_y, marginal_x="histogram", marginal_y="histogram")
        fig.write_image(outputpath+'/plots/scatter_y/' + col.replace('/', '')+'-target.webp')
    
    # box or violine plot, after normalization
    
    # C based distribution
    
    #
    # build model
    available_models_classification = [{'label': 'Baseline', 'value': 'baseline'},
                                       {'label': 'KNeighbors', 'value': 'knn'},
                                       {'label': 'SVC', 'value': 'svc'},
                                       {'label': 'GaussianProcess', 'value': 'gaussianprocess'},
                                       {'label': 'DecisionTree', 'value': 'decisiontree'},
                                       {'label': 'RandomForest', 'value': 'randomforest'},
                                       {'label': 'MLP', 'value': 'mlp'},
                                       {'label': 'AdaBoost', 'value': 'adabbost'},
                                       {'label': 'GaussianNB', 'value': 'gaussian-nb'},
                                       {'label': 'QuadraticDiscriminantAnalysis', 'value': 'qda'}]
    
    available_models_regressor = [{'label': 'Baseline', 'value': 'baseline'},
                                  {'label': 'Linear', 'value': 'linear'},
                                  {'label': 'SVR', 'value': 'svr'},
                                  {'label': 'SVR-Poly', 'value': 'svr-poly'},
                                  {'label': 'SVR-Sigmoid', 'value': 'svr-sigmoid'},
                                  {'label': 'GaussianProcess', 'value': 'gaussianprocess'},
                                  {'label': 'GaussianProcess-dw', 'value': 'gaussianprocess-dw'},
                                  {'label': 'DecisionTree', 'value': 'decisiontree'},
                                  {'label': 'RandomForest', 'value': 'randomforest'},
                                  {'label': 'MLP', 'value': 'mlp'},
                                  {'label': 'AdaBoost', 'value': 'adaboost'}]
    
    if art == 'C':
        available_models = available_models_classification
        
    elif art == 'R':
        available_models = available_models_regressor
        
    else:
        print(f'Unrecognized art {art}, use regression instead')
        available_models = available_models_regressor
    
    ## dash layout
    app.layout = html.Div([        
        add_title(),
        
        # add statistic report
        html.H2(f"Statistic description of the numeric features"),
        add_dataframe_table(dat_describe, width = '1000px', height = '300px'),
        html.Hr(),
        
        # add data check report
        html.H2(f"Check data"),
        add_checkdata_information(),
        add_dataframe_table(dat_number_na, width = '300px', height = '100px'),
        html.Hr(),
        
        # add data process report
        html.H2(f"Process data"),
        add_processdata_information(),
        html.Hr(),
        
        # add feature exploration information
        html.H2(f"Feature exploration"),
        
        html.H4(f"feature importance"),
        add_feature_importance_information(fig_feature_importances, feature_importances, width = '600px', height = '400px'),
        
        html.H4(f"Correlation between features"),
        add_heatmap_information(fig_heatmap, tmp_heatmap),
        html.Hr(),
        
        html.H4(f"Scatter between two features"),
        ####TODO!!!!: create function to select the interesting feature pair, otherwise, our customer have no idea which features should he pay attention to and why.
        dcc.Dropdown(
            id = "dropdown_scatter_features1",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[0],
            multi = False,
            clearable=False,
        ),
        dcc.Dropdown(
            id = "dropdown_scatter_features2",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[0],
            multi = False,
            clearable=False,
        ),
        dcc.Graph(id="figure_scatter_features"),
        html.Hr(),
        
        html.H4("Scatter between features and target"),
        dcc.Dropdown(
            id = "dropdown_scatter_target",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[0],
            multi = False,
            clearable = False,
        ),
        dcc.Graph(id="figure_scatter_target"),
        ###TODO!!!!: add information about the explanation of the scatter plot, about how to read scatter plot
        html.Hr(),
        
        html.H4("Violin distribution of the important features after normalization"),
        add_violin_information_of_important_features(tmp_dat),
        dcc.Dropdown(
            id = "dropdown_violin_features",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[:2],
            multi = True,
        ),
        dcc.Graph(id="figure_violin_features"),
        html.Hr(),
        
        html.H4("Class based violin distribution of the important features"),
        ###TODO!!!!: use distribution similarity to choose the interesting feature
        dcc.Dropdown(
            id = "dropdown_class_based_violin_feature",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[0],
            multi = False,
        ),
        dcc.Graph(id="figure_class_based_violin_feature"),
        html.Hr(),
        
        html.H4("Comparison of basic model performance"),
        ###TODO!!!!: add dropdown to select the target metric. maybe use checklist instead of dropdown
        dcc.Dropdown(
            id = "dropdown_basic_model_comparison",
            options = available_models,
            value = dat.columns[:2],
            multi = True
        ),
        dcc.Graph(id = "figure_basic_model_comparison"),
        html.Hr(),
    ])
    
    @app.callback(Output('figure_scatter_features', 'figure'), 
                  [Input('dropdown_scatter_features1', 'value'), Input('dropdown_scatter_features2', 'value')])
    def _update_scatter_plot_features(feature1, feature2):
        """update scatter plot: use global feature: dat"""
        out = px.scatter(dat, x=feature1, y=feature2, marginal_x="histogram", marginal_y="histogram")
        return out
    
    @app.callback(Output('figure_scatter_target', 'figure'),
                 [Input('dropdown_scatter_target', 'value')])
    def _update_scatter_plot_target(feature):
        """update scatter plot: use global features: dat, dat_y"""
        out = px.scatter(dat, x=feature, y=dat_y, marginal_x='histogram', marginal_y='histogram')
        return out
    
    @app.callback(Output('figure_violin_features', 'figure'),
                 [Input('dropdown_violin_features', 'value')])
    def _update_violin_plot_features(values):
        out = create_violin_information_of_important_features(dat[values])
        return out
    
    @app.callback(Output('figure_class_based_violin_feature', 'figure'),
                 [Input('dropdown_class_based_violin_feature', 'value')])
    def _update_class_based_violin_plot_features(value):
        out = create_class_based_violin_information_of_important_features(dat, value, dat_y, le_name_mapping)
        return out
    
    @app.callback(Output('figure_basic_model_comparison', 'figure'),
                 [Input('dropdown_basic_model_comparison', 'value')])
    def _update_basic_model_comparison(values):
        ####TODO: this process maybe slow, maybe run all and saved locally instead
        dat_comparison, fig_comparison = basic_model_comparison_classification(dat, dat_y, values)
        
        dat_comparison.to_csv(outputpath + '/performance_comparison.csv')
        fig_comparison.write_image(outputpath+'/plots/performance_comparison.webp')
        
        return fig_comparison
        
    # run server
    app.run_server()

def add_title():
    """
    create the title of the report maki use of global variables: art, targetname, sep, outputpath, datapath
    """
    out = html.Div([
        # title
        html.H1(f"Data analyse report for dataset {datapath.split('/')[-1].split('.')[0]}",
        style={
            'textAlign': 'center'
        }),
        
        # sub information
        html.Div([
            html.Label(f"Art of Task: {'Classification' if art == 'C' else 'Regression'},\t"),
            html.Label(f"Name of Target: {targetname},\t"),
            html.Label(f"Seperator: {sep},\t"),
            html.Label(f"Path of Output file: {outputpath},\t")
        ],
            style={
                'textAlign': 'center',
                'columncount': 2
        }),
        
        html.Hr()
    ])
    
    return out
    
def add_dataframe_table(df: pd.DataFrame, width: str, height: str):
    """
    display a dataframe with dash
    =============
    Parameter
    =============
    df, type of DataFrame
        the target dataframe to display with dash
    width, type of str
        the width of the displayed table, in form of number+px, e.g., 100px
    height, type of str
        the height of the displayed table
    """
    out = html.Div([        
        dash.dash_table.DataTable(df.to_dict('records'),[{'name': i, 'id': i} for i in df.columns]),
        
    ],
    style = {"overflow": "scroll", "width": width, "height": height})
    return out

def add_checkdata_information():
    """
    load the text file saved with the checkdata function and display them to the dash
    use the global variable outputpath and the magic information 'name of check data file'
    """
    # read report
    file = open(outputpath + "/data_report.txt", 'r')
    report = file.read()
    file.close
    
    # add to dash
    out = dcc.Markdown(report)
    
    return out
    
def add_processdata_information():
    """
    fill na/inf value, process date data and encoding 
    """
    file = open(outputpath + "/data_process.txt", 'r')
    report = file.read()
    file.close
    
    # add to dash
    out = dcc.Markdown(report)
    
    return out

def add_feature_importance_information(fig, df: pd.DataFrame, width: str = '800px', height: str = '400px'):
    """
    display the bar plot and explain the bar plot with the information given by the dataframe
    """
    out = html.Div([html.Div([dcc.Graph(id = 'feature importances', figure = fig, 
                                  style = {'overflow': 'scroll', 'width': width, 'height': height}),
                              add_dataframe_table(df, width = width, height=height)],
                              style = {'columnCount': 2}),
                    html.P(f'The above is the importance of the features obtained from the random forest, and it shows the importance of individual attributes for target prediction. Meaning of the value are still missing, More description needed~~~~~~~~')
                   ])
    
    return out
    
def add_heatmap_information(fig, df: pd.DataFrame, width: str = '600px', height: str = '400px'):
    """
    show the heatmap of the target dataframe
    """
    # further analyse
    df_meaning_heatmap = pd.DataFrame([['0.8-1.0', 'very strong'], ['0.6-0.8', 'strong'], ['0.4-0.6', 'middle'], ['0.2-0.4', 'weak'], ['0.0-0.2', 'very weak/no relation']], columns = ['Range (absolute)', 'Strongness of correlation'])
    
    # extract the strong and very strong correlation
    tmp_10 = df.reset_index().melt(id_vars='index').query('value == 1').query('index != variable')
    tmp_8_10 = df.reset_index().melt(id_vars='index').query('(value >=0.8)&(value<1)')
    tmp_6_8 = df.reset_index().melt(id_vars='index').query('(value >=0.6)&(value<.8)')
    tmps = []
    
    def _add_x_corr(df, x = '100'):
        if df.shape[0]==0:
            return None
        
        out = html.Div([
            html.P(f'Following feature paar are ***{x}%*** corelated:'),
            add_dataframe_table(df, width = '600px', height = '100px'),
        ])
        
        return out
    
    # plot
    out = html.Div([
              dcc.Graph(id = 'heatmap', figure = fig, style = {'width': width, 'height': height}),
        
              html.Div([html.P(f'The heatmap shows the relationship between two features (includes the extended features) in the given dataset. The correlation value range in ***[-1, 1]***. The meaning of the value is shown below:'),
                  add_dataframe_table(df_meaning_heatmap, width = '600px', height = '200px')]),
              _add_x_corr(tmp_10, '100'),
              _add_x_corr(tmp_8_10, '80'),
              _add_x_corr(tmp_6_8, '60')
    
          ], style = {'columnCount': 2})
    
    return out

def add_violin_information_of_important_features(df):
    fig = create_violin_information_of_important_features(df)
    out = dcc.Graph(figure = fig)
    
    return out

def add_class_based_violin_information_of_important_features(df, col, df_y, le_name_mapping):
    fig = create_class_based_violin_information_of_important_features(df, col, df_y, le_name_mapping)
    out = None
    
    if fig:
        out = dcc.Graph(figure = fig)
        
    return out

def create_violin_information_of_important_features(df):
    """
    create violin plot for ten most important features,
    use the global features: tmp_dat, outputpath
    """
    tmp = (df-df.mean())/df.std()
    tmp2 = []

    for i in tmp.columns:
      tmp3 = pd.DataFrame(columns = ['value', 'fname'])
      tmp3['value'] = tmp[i].values
      tmp3['fname'] = i#[i for j in range(tmp3.shape[0])]

      tmp2.append(tmp3)

    tmp = pd.concat(tmp2, axis = 0)

    fig = go.Figure()
    for i in df.columns:
        fig.add_trace(go.Violin(y=tmp['value'][tmp['fname'] == i], x= tmp['fname'][tmp['fname'] == i],
              name = i,
              box_visible=True, 
              #line_color='black',
              meanline_visible=True, #fillcolor='lightseagreen', 
              #opacity=0.6
              ))
    
    fig.write_image(outputpath+'/plots/violin_features.webp')
        
    return fig
    
def create_class_based_violin_information_of_important_features(df, col, df_y, le_name_mapping):
    """
    create figure distribution of feature for each class
    only work for classification task, if type "C" return figure else return None
    make use of global variables: le_name_mapping
    """
    inv_le_name_mapping = {}
    for i, j in le_name_mapping.items():
        inv_le_name_mapping[j] = i
    
    fig = None
    
    if art == 'C':
        fig = go.Figure()

        for i in inv_le_name_mapping.keys():

            fig.add_trace(go.Violin(y=df[col][df_y == i], x= pd.Series(df_y[df_y == i]).map(lambda x: inv_le_name_mapping[x]),
                name = inv_le_name_mapping[i],
                box_visible=True, 
                points='all',
                #line_color='black',
                meanline_visible=True, #fillcolor='lightseagreen', 
                #legendgroup='group',
                showlegend=True
                ))

        fig.write_image(outputpath+'/plots/class_based_distribution/' + col.replace('/', '') + '.webp')
    
    return fig
 


if __name__ == '__main__':
    main()
    #app.run_server()

