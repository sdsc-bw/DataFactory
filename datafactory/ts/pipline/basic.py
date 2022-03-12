# dash
import dash

from dash import dcc, html
from dash.dependencies import Input, Output
import dash_interactive_graphviz

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

sys.path.append('../preprocessing')
from ..preprocessing.encoding import * # methods for encoding
from ..preprocessing.outlier_detecting import outlier_detection_feature, outlier_detection_dataframe # methods for outlier detection
from ..preprocessing.cleaning import * # methods for data cleaning
from ..preprocessing.sampling import * # mehtods for sampling
from ..preprocessing.validating import * # methods for data checking
from ..preprocessing.loading import *
from ..preprocessing.model_comparison import basic_model_comparison
from ..plotting.model_plotting import compute_fig_from_df, plot_feature_importance_of_random_forest, plot_decision_tree # plot method


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

sys.path.append('../../util')
from ...util.constants import logger

## Setup dash
app = dash.Dash(__name__)
#app.layout = html.H1('This is my first DASH Application')

#### ATTENTION-IDEA: maybe, instead of the waterflow layout, use dash Tabs()+Tab() to create a better display

def run_pipline(data_type: str, file_path: str, output_path='./report', model_type='C', sep=',', index_col: Union[str, int]=0, header: str='infer', target_col='target'):
    # check dir existent
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
     
    if not os.path.isdir(output_path + '/plots/'):
        os.mkdir(output_path + '/plots/')
    
    if not os.path.isdir(output_path + '/plots/scatter_att/'):
        os.mkdir(output_path + '/plots/scatter_att/')
    
    if not os.path.isdir(output_path + '/plots/scatter_y/'):
        os.mkdir(output_path + '/plots/scatter_y/')
    
    if not os.path.isdir(output_path + '/plots/class_based_distribution/'):
        os.mkdir(output_path + '/plots/class_based_distribution/')
    
    # load data
    # TODO add other parameters
    df = load_dataset_from_file(data_type, file_path, sep=sep, index_col=index_col)
    
    #### check data
    # check data shape 
    if df.shape[1] < 2:
        print(f'The number of features found in the dataset is {df.shape[2]}, it may due to the wrong setting of seperator, please rerun the programm and set the seperator with parameter --sep')
    
    # statistic information
    dat_describe = df.describe()
    dat_describe.to_csv(output_path + '/statistic.csv')
    dat_describe = pd.read_csv(output_path + '/statistic.csv')
    
    # check data, 
    infofile = open(output_path + "/data_report.txt", "w")
    dat_numeric, dat_categoric, dat_y, dat_number_na, le_name_mapping, flag_balance, flag_wrong_target = check_data_and_distribute(df, model_type='C', file = infofile, logger = logger, target_col=target_col)
    infofile.close()
    dat_number_na.to_csv(output_path + '/number_na.csv')
    
    dat_number_na = pd.read_csv(output_path + '/number_na.csv')
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
    infofile = open(output_path + "/data_process.txt", "w")
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
    outlier.to_csv(output_path + '/outlier.csv')
    
    # show feature importance
    fig_feature_importances, feature_importances = plot_feature_importance_of_random_forest(dat, dat_y)#, strategy = 'permutation')
    fig_feature_importances.write_image(output_path + '/plots/feature_importance.webp')
    feature_importances.to_csv(output_path + '/feature_importances.csv')
    feature_importances = pd.read_csv(output_path + '/feature_importances.csv')
    feature_importances.columns = ['feature', 'importance']
    
    # show heatmap
    tmp_heatmap = dat.corr()
    fig_heatmap = px.imshow(tmp_heatmap)
    fig_heatmap.update_xaxes(side="bottom")
    fig_heatmap.write_image(output_path + '/plots/correlation_heatmap.webp')
    
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
    #        fig.write_image(output_path+'/plots/scatter_att/' + tmp_dat.columns[i].replace('/', '')+'-'+tmp_dat.columns[j].replace('/', '')+'.webp')
    
    # show relationship between attributes and y
    for col in dat.columns:
        fig = px.scatter(dat, x=col, y=dat_y, marginal_x="histogram", marginal_y="histogram")
        fig.write_image(output_path + '/plots/scatter_y/' + col.replace('/', '') + '-target.webp')
    
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
    
    if model_type == 'C':
        available_models = available_models_classification
        metrics = ['accuracy', 'average_precision', 'f1_weighted', 'roc_auc']
    elif model_type == 'R':
        available_models = available_models_regressor
        metrics = ['explained_variance', 'max_error', 'neg_mean_absolute_error','neg_mean_squared_error','r2']
    else:
        print(f'Unrecognized model_type {model_type}, use regression instead')
        available_models = available_models_regressor
        
    dat_comparison, dt = basic_model_comparison(dat, dat_y, available_models, metrics, model_type=model_type)
    fig_comparison = compute_fig_from_df(model_type, dat_comparison, metrics)
    dat_comparison.to_csv(output_path + '/performance_comparison.csv')
    fig_comparison.write_image(output_path + '/plots/performance_comparison.webp')
    dt_graph, dt_viz = plot_decision_tree(dt, dat, dat_y) 
    dt_viz.save(output_path + "/plots/dt_visualization.svg")
    
    ## dash layout
    app.layout = html.Div([        
        add_title(data_type, file_path, output_path=output_path, model_type=model_type, sep=sep),
        
        # add statistic report
        html.H2(f"Statistical Description"),
        html.P('Here are the common statistical measurements applied on the numeric features of the dataset.'),
        add_dataframe_table(dat_describe),
        html.Hr(),
        
        # add data check report
        html.H2(f"Check Data"),
        add_checkdata_information(output_path=output_path),
        add_dataframe_table(dat_number_na, width = '300px', height = '100px'),
        html.Hr(),
        
        # add data process report
        html.H2(f"Clean data"),
        add_processdata_information(output_path=output_path),
        html.Hr(),
        
        # add feature exploration information
        html.H2(f"Feature Exploration"),
        
        html.H4(f"Feature Importance", style={"text-decoration": "underline"}),
        html.P(f'The importance of the features is obtained from a random forest. It shows the importance of individual attributes for target prediction. The importance of a feature is between [0, 1]. The higher the importance, the higher is the influence of the feature to the target prediction.'),
        add_feature_importance_information(fig_feature_importances, feature_importances, width = '600px', height = '400px'),
        
        # ERROR HERE
        html.H4(f"Correlation between features", style={"text-decoration": "underline"}),
        html.P(f'The heatmap shows the relationship between two features (includes the extended features) in the given dataset. The correlation value range in [-1, 1]. A negative correlation means that the relation ship between two features in which one variable increases as the other decreases. The meaning of the values is shown below:'),
        add_heatmap_information(fig_heatmap, tmp_heatmap),
        
        html.H4(f"Scatter Between Two Features", style={"text-decoration": "underline"}),
        html.P("This scatter plot displays the values of two features of the dataset. It can show the degree of the correlation between two features. If the points' pattern slopes from lower left to upper right, it indicates a positive correlation. If the pattern of points slopes from upper left to lower right, it indicates a negative correlation."),
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
        
        html.H4("Scatter Between Features and Target", style={"text-decoration": "underline"}),
        html.P("This scatter plot displays the values of a selected feature with the target."),
        dcc.Dropdown(
            id = "dropdown_scatter_target",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[0],
            multi = False,
            clearable = False,
        ),
        dcc.Graph(id="figure_scatter_target"),
        
        html.H4("Violin Distribution of the Important Features after Normalization", style={"text-decoration": "underline"}),
        html.P("This violin plot shows the probability density of the data at the selected features. It also contains a marker for the statistical metrics above."),
        dcc.Dropdown(
            id = "dropdown_violin_features",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[:2],
            multi = True,
        ),
        dcc.Graph(id="figure_violin_features"),
        
        html.H4("Class-based Violin Distribution of the Important Features", style={"text-decoration": "underline"}),
        ###TODO!!!!: use distribution similarity to choose the interesting feature
        html.P("This violin plot shows the probability density of the data at the important features."),
        dcc.Dropdown(
            id = "dropdown_class_based_violin_feature",
            options = [{'label': col, 'value': col} for col in dat.columns],
            value = dat.columns[0],
            multi = False,
        ),
        dcc.Graph(id="figure_class_based_violin_feature"),
        html.Hr(),
        
        html.H2("Comparison of Basic Models"),
        html.P("Here we can see how the basic machine learning models peform on the task. Use the checkboxes to select the metric of the comparison."),
        ###TODO!!!!: add dropdown to select the target metric. maybe use checklist instead of dropdown
        dcc.Dropdown(
            id = "dropdown_basic_model_comparison",
            options = available_models,
            value = dat.columns[:2],
            multi = True
        ),
        dcc.Graph(id = "figure_basic_model_comparison"),
        html.Hr(),
        
        html.H2("Decision Tree Visualization"),
        html.P("Here we can see the visualization of a decision tree."),
        dash_interactive_graphviz.DashInteractiveGraphviz(id="dt_graph", dot_source=dt_graph)
    ])
    
    @app.callback(Output('table_corr_per', 'data'),
                 [Input('dropdown_corr_per', 'value')])
    def _update_table_corr(x):
        """update correlation table: use heatmap"""
        df_heatmap = tmp_heatmap.reset_index().melt(id_vars='index').query(f'(value >={int(x)/100})&(value<1)') 
        out = df_heatmap.to_dict('records')
        return out
    
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
        out = create_violin_information_of_important_features(dat[values], output_path=output_path)
        return out
    
    @app.callback(Output('figure_class_based_violin_feature', 'figure'),
                 [Input('dropdown_class_based_violin_feature', 'value')])
    def _update_class_based_violin_plot_features(value):
        out = create_class_based_violin_information_of_important_features(dat, value, dat_y, le_name_mapping, output_path=output_path, model_type=model_type)
        return out
    
    @app.callback(Output('figure_basic_model_comparison', 'figure'),
                 [Input('dropdown_basic_model_comparison', 'value')])
    def _update_basic_model_comparison(values):
        #print(dat_comparison)
        print(fig_comparison)
        
        return fig_comparison
        
    # run server
    app.run_server()

def add_title(data_type: str, file_path: str, output_path='./report/', model_type='C', sep=','):
    """
    create the title of the report maki use of global variables: model_type, sep, output_path, datapath
    """
    out = html.Div([
        # title
        html.H1(f"Data Analyse for Dataset: {file_path.split('/')[-1].split('.')[0]}",
        style={
            'textAlign': 'center'
        }),
        
        # sub information
        html.Div([
            html.Strong(f"Type of Task: "),
            html.Label(f"{'Classification' if model_type == 'C' else 'Regression'},\t"),
            html.Strong(f"Datatype: "),
            html.Label(f"{data_type},\t"),
            html.Strong(f"Path of Output file: "),
            html.Label(f"{output_path},\t")
        ],
            style={
                'textAlign': 'center',
                'columncount': 2
        }),
        
        html.Hr()
    ])
    
    return out
    
def add_dataframe_table(df: pd.DataFrame, id=None, width: str='auto', height: str='auto'):
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
    if id:
        out = html.Div([        
            dash.dash_table.DataTable(df.to_dict('records'),[{'name': i, 'id': i} for i in df.columns], id=id),

        ],
        style = {"overflow": "auto", "width": width, "height": height})
    else:
        out = html.Div([        
            dash.dash_table.DataTable(df.to_dict('records'),[{'name': i, 'id': i} for i in df.columns]),

        ],
        style = {"overflow": "auto", "width": width, "height": height})
    return out

def add_checkdata_information(output_path='./report'):
    """
    load the text file saved with the checkdata function and display them to the dash
    use the global variable output_path and the magic information 'name of check data file'
    """
    # read report
    file = open(output_path + "/data_report.txt", 'r')
    report = file.read()
    file.close
    
    # add to dash
    out = dcc.Markdown(report)
    
    return out
    
def add_processdata_information(output_path: './report'):
    """
    fill na/inf value, process date data and encoding 
    """
    file = open(output_path + "/data_process.txt", 'r')
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
                                  style = {'overflow': 'auto', 'width': width, 'height': height}),
                              add_dataframe_table(df, width=width, height=height)],
                              style = {'columnCount': 2})
                   ])
    
    return out

def _add_x_corr(df, x='100'):
    if df.shape[0]==0:
        return None
     
    df_heatmap = df.reset_index().melt(id_vars='index').query(f'(value >={int(x)/100})&(value<1)')
        
    ticks = np.arange(1, 100)
        
    out = html.Div([
            html.Div([
                html.P(f'Following feature pairs are over', style={'margin-right': '10px'}),
                dcc.Dropdown(
                    id="dropdown_corr_per",
                    options=[{'label': x , 'value': x} for x in ticks],
                    value = 80, 
                    multi=False,
                    style={
                        'width':'40px',
                        'height':'30px',
                        'display':'inline-block',
                        'margin-top':'3px',
                        'verticalAlign':"middle"
                    },
                    clearable=False
                ),
                html.P(f'%'),
                html.P(f'correlated:', style={'margin-left': '10px'}),
            ], style=dict(display='flex')),
            add_dataframe_table(df_heatmap, id='table_corr_per', width = '600px', height = '200px'),
        ])
    return out
    
def add_heatmap_information(fig, df: pd.DataFrame, width: str = '600px', height: str = '400px'):
    """
    show the heatmap of the target dataframe
    """
    # further analyse
    df_meaning_heatmap = pd.DataFrame([['0.8-1.0', 'very strong'], ['0.6-0.8', 'strong'], ['0.4-0.6', 'middle'], ['0.2-0.4', 'weak'], ['0.0-0.2', 'very weak/no relation']], columns = ['Range (absolute)', 'Strongness of correlation'])
    
    # extract the strong and very strong correlation
    #tmp_10 = df.reset_index().melt(id_vars='index').query('value == 1').query('index != variable')
    #tmp_8_10 = df.reset_index().melt(id_vars='index').query('(value >=0.8)&(value<1)')
    #tmp_6_8 = df.reset_index().melt(id_vars='index').query('(value >=0.6)&(value<.8)')
    #tmps = []
    
    # plot
    out = html.Div([
              dcc.Graph(id = 'heatmap', figure = fig, style = {'width': width, 'height': height}),
        
              add_dataframe_table(df_meaning_heatmap, width = '600px', height = '200px'),
              _add_x_corr(df, '80'),
    
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

def create_violin_information_of_important_features(df, output_path='./report'):
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
    
    fig.write_image(output_path + '/plots/violin_features.webp')
        
    return fig
    
def create_class_based_violin_information_of_important_features(df, col, df_y, le_name_mapping, output_path='./report', model_type='C'):
    """
    create figure distribution of feature for each class
    only work for classification task, if type "C" return figure else return None
    make use of global variables: le_name_mapping
    """
    inv_le_name_mapping = {}
    for i, j in le_name_mapping.items():
        inv_le_name_mapping[j] = i
    
    fig = None
    
    if model_type == 'C':
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

        fig.write_image(output_path + '/plots/class_based_distribution/' + col.replace('/', '') + '.webp')
    
    return fig
