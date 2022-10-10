import pandas as pd #Package zur Tabellenberechnung
from datetime import timedelta #Package zur Verwendung von Daten

import graphviz
from dtreeviz.trees import dtreeviz

from datetime import datetime, timedelta #Package zur Verwendung von Daten

# Packages um Website und Plots zu generieren
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State

#datafactory
import sys
sys.path.append('../preprocessing')
from ..preprocessing.loading import *
from ..preprocessing.splitting import *
from ..preprocessing.encoding import * # methods for encoding
from ..preprocessing.outlier_detecting import outlier_detection_feature, outlier_detection_dataframe # methods for outlier detection
from ..preprocessing.cleaning import * # methods for data cleaning
from ..preprocessing.validating import * # methods for data checking
from ..plotting.model_plotting import compute_fig_from_df

sys.path.append('../model_training')
from ..model_training.basic_model_training import compare_models

sys.path.append('../model_explaining')
from ..model_explaining.model_explaining import explain_models

sys.path.append('../../util')
from ...util.constants import logger
from ...util.models import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = DashProxy(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])

def run_pipline(data_type: str, model_type, file_path, is_file=True, query: Union[list,str]="""select *""", output_path='./output/', 
                target_feature=None, initial_features=None, sep=',', header: str='infer', index_col: Union[str, int]=0,
                time_col: Union[str, List]='Time', time_format=None,
                pref=None):
    
    input_df = load_dataset(data_type, file_path, is_file=is_file, sep=sep, index_col=index_col, time_col=time_col, time_format=time_format, 
                            pref=pref)
    
    create_layout(input_df)
    
    app.run_server()

##################### Layout ######################### 

def create_layout(input_df):
    #global APP
    app.layout = html.Div([
        add_title(),
        ############## copy this to create a new row ########################
        html.Div([
            add_histogram(input_df),
            add_correlation_tile(input_df),
        ], className="row"),
        ############## copy this to create a new row ########################
        html.Div([
            add_scatter_plot_one_feature(input_df),
            add_scatter_plot_two_features(input_df),           
        ], className="row"),
        html.Div([
            add_scatter_plot_one_feature(input_df, feature='NMHC(GT)', id_postfix='_nmhc'),
            add_scatter_plot_one_feature(input_df, feature='NO2(GT)', id_postfix='_n02'),           
        ], className="row"), 
    ])
    
def add_title():
    out = html.Div([
        html.H1(f"Data Analysis"),
        html.Hr()
    ])
    
    return out

def add_correlation_tile(df, cols=None, id_postfix=''):
    """
    display a correlation matrix
    =============
    Parameter
    =============
    df, type of DataFrame
        the dataframe to create the correlation matrix from
    cols, type of List
        predefined features to show in the correlation matrix
    id_postfix, type of str
        additional string for id, if multiple tiles of this type are used
    """
    df_heatmap_all = df.corr()
    if cols is None:
        cols = df.columns[:4]
        
    fig = px.imshow(df[cols].corr())
    
    out = html.Div([
        html.H3('Correlation Matrix'),
        dcc.Dropdown(id = 'dropdown_corr_mat' + id_postfix, options=df.columns, value=cols, multi=True),
        dcc.Graph(id='corr_mat' + id_postfix, figure=fig),
    ], className='tile')
    
    @app.callback(Output('corr_mat' + id_postfix, 'figure'),
                 Input('dropdown_corr_mat' + id_postfix, 'value'))
    def update_corr_mat(cols):
        fig = px.imshow(df[cols].corr())
        return fig
    
    return out

def add_scatter_plot_one_feature(df, feature=None, id_postfix=''):
    """
    display a scatter plot of one feature
    =============
    Parameter
    =============
    df, type of DataFrame
        the dataframe to create the scatter plot from
    feature, type of List
        predefined feature to show in the scatter plot
    id_postfix, type of str
        additional string for id, if multiple tiles of this type are used
    """
    options = [{'label': col, 'value': col} for col in df.columns]
    if feature is None:
        feature = df.columns[0]
        
    fig = px.scatter(df, x=feature, marginal_x="histogram", marginal_y="histogram")
    
    out = html.Div([
        html.H3('Scatter Plot'),
        dcc.Dropdown(id="dropdown_scatter_one_feature" + id_postfix, options=options, value=feature, multi=False, clearable=False),
        dcc.Graph(id='scatter_one_feature' + id_postfix, figure=fig),
    ], className='tile')
    
    @app.callback(Output('scatter_one_feature' + id_postfix, 'figure'), 
                  Input('dropdown_scatter_one_feature' + id_postfix, 'value'))
    def update_scatter_plot_features(feature):
        fig = px.scatter(df, x=feature, marginal_x="histogram", marginal_y="histogram")
        return fig
    
    return out

def add_scatter_plot_two_features(df, feature_1=None, feature_2=None, id_postfix=''):
    """
    display a scatter plot of two features
    =============
    Parameter
    =============
    df, type of DataFrame
        the dataframe to create the scatter plot from
    feature_1, type of List
        predefined feature 1 to show in the scatter plot
    feature_2, type of List
        predefined feature 2 to show in the scatter plot
    id_postfix, type of str
        additional string for id, if multiple tiles of this type are used
    """
    options = [{'label': col, 'value': col} for col in df.columns]
    if feature_1 is None:
        feature_1 = df.columns[0]
    if feature_2 is None:
        feature_2 = df.columns[1]
        
    fig = px.scatter(df, x=feature_1, y=feature_2, marginal_x="histogram", marginal_y="histogram")
    
    out = html.Div([
        html.H3('Scatter Plot'),
        dcc.Dropdown(id="dropdown_scatter_feature_1" + id_postfix, options=options, value=feature_1, multi=False, clearable=False),
        dcc.Dropdown(id="dropdown_scatter_feature_2" + id_postfix, options=options, value=feature_2, multi=False, clearable=False),
        dcc.Graph(id='scatter_two_features' + id_postfix, figure=fig),
    ], className='tile')
    
    @app.callback(Output('scatter_two_features' + id_postfix, 'figure'), 
                  Input('dropdown_scatter_feature_1' + id_postfix, 'value'),
                  Input('dropdown_scatter_feature_2' + id_postfix, 'value'))
    def update_scatter_plot_features(feature_1, feature_2):
        fig = px.scatter(df, x=feature_1, y=feature_2, marginal_x="histogram", marginal_y="histogram")
        return fig
    
    return out

def add_histogram(df, features=None, id_postfix=''):
    """
    display a histogram
    =============
    Parameter
    =============
    df, type of DataFrame
        the dataframe to create the histogram from
    features, type of List
        predefined feature to show in the histogram
    id_postfix, type of str
        additional string for id, if multiple tiles of this type are used
    """
    options = [{'label': col, 'value': col} for col in df.columns]
    if features is None:
        features = [df.columns[0]]
        
    fig = px.histogram(df, x=features)   
    
    out = html.Div([
        html.H3('Histogram'),
        dcc.Dropdown(id="dropdown_histogram" + id_postfix, options=options, value=features, multi=True, clearable=False),
        dcc.Graph(id='histogram' + id_postfix, figure=fig),
    ], className='tile')
    
    @app.callback(Output('histogram' + id_postfix, 'figure'), 
                  Input('dropdown_histogram' + id_postfix, 'value'))
    def update_scatter_plot_features(features):
        fig = px.histogram(df, x=features)
        return fig
        
    return out
