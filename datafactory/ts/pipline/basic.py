from typing import cast, Any, Dict, List, Tuple, Optional, Union

import os

#datafactory
import sys
sys.path.append('../preprocessing')
from ..preprocessing.loading import *
from ..preprocessing.encoding import * # methods for encoding
from ..preprocessing.outlier_detecting import outlier_detection_feature, outlier_detection_dataframe # methods for outlier detection
from ..preprocessing.cleaning import * # methods for data cleaning
from ..preprocessing.validating import * # methods for data checking
from ..preprocessing.model_comparison import basic_model_comparison, get_model_with_name_regression
from ..preprocessing.exploring import compute_feature_importance_of_random_forest
from ..plotting.model_plotting import compute_fig_from_df, plot_feature_importance_of_random_forest, plot_decision_tree # plot method

sys.path.append('../../util')
from ...util.constants import logger

# dash
import dash
import matplotlib.pyplot a plt
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_interactive_graphviz

# plot packages
import plotly.express as px
import plotly.graph_objs as go

## Setup dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

def run_pipline(data_type: str, file_path: str, output_path='./report', model_type='C', sep=',', index_col: Union[str, int]=0, header: str='infer', target_col='target'):
    global FILE_PATH, OUTPUT_PATH, MODEL_TYPE
    FILE_PATH = file_path
    OUTPUT_PATH = output_path
    MODEL_TYPE = model_type
    
    # create directories for the report
    _create_output_directory(output_path)
    
    # load dataset
    # TODO add other parameters
    df = load_dataset_from_file(data_type, file_path, sep=sep, index_col=index_col)
    
    # basic information
    _get_statistical_information(output_path, df)
    _check_data(output_path, target_col, df, model_type)
    _get_outlier(output_path, X)
    
    # feature distibution
    _get_violin_distribution(X, output_path=output_path)
    
    # correlation
    _get_corr_heatmap(output_path, X)
    
    # feature importance
    global FEATURE_IMPORTANCE
    FEATURE_IMPORTANCE = _get_feature_importance(X, Y, model_type)
    
    # decision tree and model comparison
    global AVAILABLE_MODELS, METRICS
    AVAILABLE_MODELS, METRICS = _get_available_models_and_metrics(model_type)
    _get_dt_and_model_comparison(output_path, X, Y, model_type, AVAILABLE_MODELS, METRICS)
    
    create_layout()
    
    add_callbacks()
    
    app.run_server()
    
def _create_output_directory(output_path):
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
        
def _check_data(output_path, target_col, df, model_type):
    if df.shape[1] < 2:
        logger.warn(f'The number of features found in the dataset is {df.shape[2]}, it may due to the wrong setting of seperator, please rerun the programm and set the seperator with parameter \'sep=...\'')
        
    global DATA_NUMERIC, DATA_CATEGORIC, Y, N_NUMERIC_NAN, LE_NAME_MAPPING
    info_file = open(output_path + "/data_report.txt", "w")
    DATA_NUMERIC, DATA_CATEGORIC, Y, N_NUMERIC_NAN, LE_NAME_MAPPING, _, flag_wrong_target = check_data_and_distribute(df, model_type=model_type, file=info_file, target_col=target_col)
    info_file.close()
    
    N_NUMERIC_NAN.to_csv(output_path + '/number_nan.csv')
    N_NUMERIC_NAN = pd.read_csv(output_path + '/number_nan.csv')
    N_NUMERIC_NAN.columns = ['feature', '#NAN']
    N_NUMERIC_NAN = N_NUMERIC_NAN[N_NUMERIC_NAN['#NAN'].map(lambda x: float(x.split('/')[0])/float(x.split('/')[1])) > 0]
    
    inp = 'yes'
    if flag_wrong_target:
        # TODO rework error message
        inp = input("There are more then 10 classed in the given dataset that contain less than 10 items. We suggestion that the given target may be wrong, please check the data: return 'yes' to continue: ")
        
    if inp == 'yes':
        logger.info(f'Continue with the given target: {target_col}')
    else:
        raise ValueError('Input the correct target.')
        
    info_file = open(output_path + "/data_process.txt", "w")
    DATA_NUMERIC = clean_data(DATA_NUMERIC, file = info_file)
    
    DATA_CATEGORIC = categorical_feature_encoding(DATA_CATEGORIC, file = info_file)
    
    global X
    X = pd.concat([DATA_NUMERIC, DATA_CATEGORIC], axis=1)
    global DF
    DF = pd.concat([X, pd.DataFrame(Y, columns = ['target'])], axis=1)
    
    logger.info(f'Shape of the dataframe after processing is ***{X.shape}***')
    
    info_file.close()
        
def _get_outlier(output_path, df):
    OUTLIER = outlier_detection_dataframe(df) # maybe should use dat_numeric as input instead of the whole dat
    OUTLIER.to_csv(output_path + '/outlier.csv')
        
def _get_statistical_information(output_path, df):
    global DATA_DESCRIPTION
    DATA_DESCRIPTION = df.describe()
    DATA_DESCRIPTION.to_csv(output_path + '/statistic.csv')
    DATA_DESCRIPTION = pd.read_csv(output_path + '/statistic.csv')
    
def _get_corr_heatmap(output_path, df):
    global DF_HEATMAP, FIG_HEATMAP
    DF_HEATMAP = df.corr()
    FIG_HEATMAP = px.imshow(DF_HEATMAP)
    FIG_HEATMAP.update_xaxes(side="bottom")
    FIG_HEATMAP.write_image(output_path + '/plots/correlation_heatmap.webp')

def _get_available_models_and_metrics(model_type):
    available_models_classification = [{'label': 'Baseline', 'value': 'baseline'},
                                       {'label': 'KNeighbors', 'value': 'knn'},
                                       {'label': 'SVC', 'value': 'svc'},
                                       #{'label': 'GaussianProcess', 'value': 'gaussianprocess'},
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
                                  #{'label': 'GaussianProcess-dw', 'value': 'gaussianprocess-dw'},
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
        available_models = available_models_regressor
        
    return available_models, metrics

def _get_violin_distribution(df, output_path=None):
    # TODO move to df + split in computation and visualization
    tmp = (df-df.mean())/df.std()
    tmp2 = []

    for i in tmp.columns:
        tmp3 = pd.DataFrame(columns = ['value', 'fname'])
        tmp3['value'] = tmp[i].values
        tmp3['fname'] = i#[i for j in range(tmp3.shape[0])]

        tmp2.append(tmp3)

    violin_distribution = pd.concat(tmp2, axis=0)
    
    fig = go.Figure()
    for i in df.columns:
        fig.add_trace(go.Violin(y=violin_distribution['value'][violin_distribution['fname'] == i], x= violin_distribution['fname'][violin_distribution['fname'] == i],
              name = i,
              box_visible=True, 
              #line_color='black',
              meanline_visible=True, #fillcolor='lightseagreen', 
              #opacity=0.6
              ))
    
    if output_path:
        fig.write_image(output_path + '/plots/violin_features.webp')
        
    return fig

def _get_class_based_violin_distribution(X, y, col, le_name_mapping, output_path=None):
    # TODO move to df + split in computation and visualization
    inv_le_name_mapping = {}
    for i, j in le_name_mapping.items():
        inv_le_name_mapping[j] = i
    fig = go.Figure()

    for i in inv_le_name_mapping.keys():
        fig.add_trace(go.Violin(y=X[col][y == i], 
                                x= pd.Series(y[y == i]).map(lambda x: inv_le_name_mapping[x]),
                                name = inv_le_name_mapping[i],
                                box_visible=True, 
                                points='all',
                                #line_color='black',
                                meanline_visible=True, 
                                #fillcolor='lightseagreen', 
                                #legendgroup='group',
                                showlegend=True))

    if output_path:
        fig.write_image(output_path + '/plots/class_based_distribution/' + col.replace('/', '') + '.webp')
    
    return fig
    
def _get_feature_importance(X, y, model_type):
    feature_importances = compute_feature_importance_of_random_forest(X, y, model_type=model_type)
    return feature_importances
    
def _get_dt_and_model_comparison(output_path, X, y, model_type, available_models, metrics):
    global MODEL_COMPARISON, DT_GRAPH
    MODEL_COMPARISON, dt = basic_model_comparison(X, y, available_models, metrics, model_type=model_type)
    fig_comparison = compute_fig_from_df(model_type, MODEL_COMPARISON, metrics)
    MODEL_COMPARISON.to_csv(output_path + '/performance_comparison.csv')
    fig_comparison.write_image(output_path + '/plots/performance_comparison.webp')
    DT_GRAPH, dt_viz = plot_decision_tree(dt, X, y) 
    dt_viz.save(output_path + "/plots/dt_visualization.svg")
    
##################################### Layout ############################################
def create_layout():
    app.layout = html.Div([
        _add_title(),
        dcc.Tabs([
            _add_info_tab(),
            _add_feature_distribution_tab(),
            _add_feature_correlation_tab(),
            _add_feature_importance_tab(),
            _add_dt_tab(),
            _add_model_comparison_tab()
            
        ])
    ])

def _add_title():
    out = html.Div([
        html.H1(f"Data Analyse for Dataset: {FILE_PATH.split('/')[-1].split('.')[0]}",
                style={
                    'textAlign': 'center'
                }),
        html.Hr()
    ])
    
    return out    
    
def _add_info_tab():
    out = dcc.Tab(label='Basic Information', children=[
        dcc.Tabs([
            __add_task_tab(),
            __add_statistics_tab(),
            __add_outlier_tab(),
            __add_preprocessing_tab()
        ])
    ])
    
    return out

def __add_task_tab():
    out = dcc.Tab(label='Task', children=[
        html.H4('Task'),
        #TODO add type of task, output_path, infow from data_report.txt
    ])
    
    return out

def __add_statistics_tab():
    out = dcc.Tab(label='Statistics', children=[
        html.H4('Statistics'),
        html.P('Here are the common statistical measurements applied on the numeric features of the dataset.', className='par'),
        add_dataframe_table(DATA_DESCRIPTION),
    ])
    
    return out

def __add_outlier_tab():
    out = dcc.Tab(label='Outlier', children=[
        html.H4('Outlier'),
        #TODO add figure
    ])
    
    return out

def __add_preprocessing_tab():
    out = dcc.Tab(label='Preprocessing', children=[
        html.H4('Preprocessing'),
        #TODO add infos from data_process.txt
    ])
    
    return out

def _add_feature_distribution_tab():
    if MODEL_TYPE == 'C':
        out = dcc.Tab(label='Feature Distribution', children=[
            dcc.Tabs([
                __add_class_distribution_tab(),
                __add_violin_distribution_tab()
            ])
        ])
    
    else:
        out = dcc.Tab(label='Feature Distribution', children=[
            dcc.Tabs([
                __add_violin_distribution_tab()
            ])
        ])
    
    return out

def __add_class_distribution_tab():
    out = dcc.Tab(label='Class distribution', children = [
        html.H4('Target Classs distribution'),
        html.P('This shows the proportion of the different classes in the total data. In principle, the more equal the proportion of different classes is, the better for the training of the model'),
        dcc.Graph(id='class_distribution', value = px.histogram(x = Y))
    ])
    return out

def __add_violin_distribution_tab():
    if MODEL_TYPE == 'C':
        out = dcc.Tab(label='Violin Distributions', children=[
            dcc.Tabs([
                ___add_violin_distribution_important_features(),
                ___add_violin_distribution_class_based(),
                ___add_violin_distribution_custom()
            ])
        ])
    else:
        out = dcc.Tab(label='Violin Distributions', children=[
            dcc.Tabs([
                ___add_violin_distribution_important_features(),
                ___add_violin_distribution_custom()
            ])
        ])
    
    return out

def ___add_violin_distribution_important_features():
    out = dcc.Tab(label='Important Features', children=[
        html.H4('Violin Distribution of Important Features'),
        html.P("This violin plot shows the probability density of the important features. It also contains a marker for the statistical metrics.", className='par'),
        dcc.Dropdown(
            id = "dropdown_violin_features",
            options = [{'label': col, 'value': col} for col in X.columns],
            value = X.columns[:2],
            multi = True,
        ),
        dcc.Graph(id="figure_violin_features"),
    ])
    
    return out

def ___add_violin_distribution_class_based():
    if MODEL_TYPE == 'C':
        out = dcc.Tab(label='Class-based', children=[
            html.H4('Class-based Violin Distribution'),
            html.P("This violin plot shows the probability density of every feature based on the classes. These classes will be useful for classification tasks if the distribution of the same attributes varies widely across classes", className='par'),
            html.Label('Feature:', className='dropdown_label'),
            dcc.Dropdown(
                id = "dropdown_class_based_violin_features",
                options = [{'label': col, 'value': col} for col in X.columns],
                value = X.columns[0],
                multi = False,
                clearable=False,
                className='dropdown',
                placeholder="Select a feature...",
            ),
            dcc.Graph(id="figure_class_based_violin_features"),
        ])
    else:
        out = None
    return out

def ___add_violin_distribution_custom():
    out = dcc.Tab(label='Custom', children=[
        html.H4('Violin Distribution of Custom Features'),
        html.P("This violin plot shows the probability density of every feature.", className='par'),
        dcc.Dropdown(
            id = "dropdown_violin_custom_features",
            options = [{'label': col, 'value': col} for col in X.columns],
            value = X.columns[:2],
            multi = True,
        ),
        dcc.Graph(id="figure_violin_custom_features"),
    ])
    
    return out
    
def _add_feature_correlation_tab():
    out = dcc.Tab(label='Feature Correlation', children=[
        dcc.Tabs([
            __add_heatmap_tab(),
            __add_scatter_plot_tab()
        ])
    ])
    
    return out

def __add_heatmap_tab():
    df_meaning_heatmap = pd.DataFrame([['0.8-1.0', 'very strong'], ['0.6-0.8', 'strong'], ['0.4-0.6', 'middle'], ['0.2-0.4', 'weak'], ['0.0-0.2', 'very weak/no relation']], columns=['Range (absolute)', 'strongness of correlation'])
    
    out = dcc.Tab(label='Heatmap', children=[
        html.H4('Heatmap of Correlation between Features'),
        html.P(f'The heatmap shows the relationship between two features (includes the extended features) in the given dataset. The correlation value range in [-1, 1]. A negative correlation means that the relation ship between two features in which one variable increases as the other decreases. The meaning of the values is shown below.', className='par'),
        html.Div([
            dcc.Graph(id='heatmap', figure=FIG_HEATMAP, className='fig_with_description'),
            add_dataframe_table(df_meaning_heatmap, className='description'),
        ])
        # TODO also add 'table_corr_per' from basic.tmp
    ])
    
    return out

def __add_scatter_plot_tab():
    out = dcc.Tab(label='Scatter Plots', children=[
        dcc.Tabs([
            ___add_scatter_plot_important_features_tab(),
            ___add_scatter_plot_important_features_target_tab(),
            ___add_scatter_plot_custom()
            
        ])
    ])
    
    return out

def ___add_scatter_plot_important_features_tab():
    # TODO select important features instead of whole X
    out = dcc.Tab(label='Important Features', children=[
        html.H4('Scatter Plots of Important Features'),
        html.P("A scatter plot displays the values of two features of the dataset. It can show the degree of the correlation between two features. If the points' pattern slopes from lower left to upper right, it indicates a positive correlation. If the pattern of points slopes from upper left to lower right, it indicates a negative correlation.", className='par'),
        html.Div([
            html.Label('X-axis:', className='dropdown_label'),
            dcc.Dropdown(
                id = "dropdown_scatter_important_features1",
                options = [{'label': col, 'value': col} for col in X.columns],
                multi = False,
                clearable=False,
                className='dropdown',
                placeholder="Select 1. feature...",
            ),
        ], className='dropdown_with_label'),
        html.Div([
            html.Label('Y-axis:', className='dropdown_label'),
            dcc.Dropdown(
                id = "dropdown_scatter_important_features2",
                options = [{'label': col, 'value': col} for col in X.columns],
                multi = False,
                clearable=False,
                className='dropdown',
                placeholder="Select 2. feature...",
            ),
        ], className='dropdown_with_label'),
        dcc.Graph(id="figure_scatter_important_features"),
        
    ])
    
    return out

def ___add_scatter_plot_important_features_target_tab():
    out = dcc.Tab(label='Features and Target', children=[
        html.H4('Scatter Plots of Important Features and Target'),
        html.P("This scatter plot displays the values of a important feature with the target y.", className='par'),
        html.Div([
            html.Label('X-axis:', className='dropdown_label'),
            dcc.Dropdown(
                id = "dropdown_scatter_target",
                options = [{'label': col, 'value': col} for col in X.columns],
                multi = False,
                clearable = False,
                className='dropdown',
                placeholder="Select a feature...",
            ),
        ], className='dropdown_with_label'),
        dcc.Graph(id="figure_scatter_target"),
    ])
    
    return out

def ___add_scatter_plot_custom():
    out = dcc.Tab(label='Custom', children=[
        html.H4('Scatter Plots of Custom Features'),
        html.P("Here you can create a scatter plot of every column in the dataset.", className='par'),
        html.Div([
            html.Label('X-axis:', className='dropdown_label'),
            dcc.Dropdown(
                id = "dropdown_scatter_custom_features1",
                options = [{'label': col, 'value': col} for col in DF.columns],
                multi = False,
                clearable=False,
                className='dropdown',
                placeholder="Select 1. feature...",
            ),
        ], className='dropdown_with_label'),
        html.Div([
            html.Label('Y-axis:', className='dropdown_label'),
            dcc.Dropdown(
                id = "dropdown_scatter_custom_features2",
                options = [{'label': col, 'value': col} for col in DF.columns],
                multi = False,
                clearable=False,
                className='dropdown',
                placeholder="Select 2. feature...",
            ),
        ], className='dropdown_with_label'),
        dcc.Graph(id="figure_scatter_custom_features"),
    ])
    
    return out


def _add_feature_importance_tab():
    # TODO edit meaning
    df_meaning_importance = pd.DataFrame([['0.8-1.0', 'very high'], ['0.6-0.8', 'high'], ['0.4-0.6', 'middle'], ['0.2-0.4', 'low'], ['0.0-0.2', 'very low']], columns=['Range (absolute)', 'importance'])
    out = dcc.Tab(label='Feature Importance', children=[
        html.H4('Feature Importance'),
        html.P(f'The importance of the features is obtained from a random forest. It shows the importance of individual attributes for target prediction. The importance of a feature is between [0, 1]. The higher the importance, the higher is the influence of the feature to the target prediction.', className='par'),
        html.Div([
            dcc.Graph(id='figure_feature_importance', figure=plot_feature_importance_of_random_forest(FEATURE_IMPORTANCE), className='fig_with_description'),
            add_dataframe_table(df_meaning_importance, className='description'),
        ])
    ])
    
    return out

def _add_dt_tab():
    # TODO do not allow scrolling with mouse (buttons instead) and change initial size (at the moment to large)
    out = dcc.Tab(label='Decision Tree', children=[
        html.H4('Visualization of a Decsion Tree'),
        dash_interactive_graphviz.DashInteractiveGraphviz(
            id="graph",
            dot_source=DT_GRAPH
        )
    ])
    
    return out

def _add_model_comparison_tab():
    if MODEL_TYPE == 'C': 
        out = dcc.Tab(label='Model Comparison', children=[
            html.H4('Performace Comparsion of Different Models'),
            html.P("Here we can see how the basic machine learning models peform on the task.", className='par'),
            dcc.Dropdown(
                id = "dropdown_basic_model_comparison",
                options = AVAILABLE_MODELS,
                value = [model['value'] for model in AVAILABLE_MODELS],
                multi = True
            ),
        ])
    
    else:
        # plot scatter for regression task
        out = dcc.Tab(label='Model Comparison', children=[
            html.H4('Performace Comparsion of Different Models'),
            html.P("Here we can see how the basic machine learning models peform on the task.", className='par'),
            dcc.Dropdown(
                id = "dropdown_basic_model_comparison",
                options = AVAILABLE_MODELS,
                value = [model['value'] for model in AVAILABLE_MODELS],
                multi = True
            ),
            dcc.Graph(id = "figure_basic_model_comparison"),
            dcc.Dropdown(
                i = "dropdown_basic_model_scatter_plot",
                options = AVAILABLE_MODELS,
                value = [model['value'] for model in AVAILABLE_MODELS],
                multi = False
            ),
            dcc.Graph(id = "figure_basic_model_scatter_plot"),
        ])
    
    return out

######################### Callbacks ######################

def add_callbacks():
    
    @app.callback(Output('figure_scatter_important_features', 'figure'), 
                  [Input('dropdown_scatter_important_features1', 'value'), Input('dropdown_scatter_important_features2', 'value')])
    def _update_scatter_plot_features(feature1, feature2):
        # TODO only use df with important features
        out = px.scatter(X, x=feature1, y=feature2, marginal_x="histogram", marginal_y="histogram")
        return out
    
    @app.callback(Output('figure_scatter_target', 'figure'),
                 [Input('dropdown_scatter_target', 'value')])
    def _update_scatter_plot_target(feature):
        # TODO only use df with important features
        out = px.scatter(X, x=feature, y=Y, marginal_x='histogram', marginal_y='histogram')
        return out
    
    @app.callback(Output('figure_scatter_custom_features', 'figure'), 
                  [Input('dropdown_scatter_custom_features1', 'value'), Input('dropdown_scatter_custom_features2', 'value')])
    def _update_scatter_plot_features(feature1, feature2):
        out = px.scatter(DF, x=feature1, y=feature2, marginal_x="histogram", marginal_y="histogram")
        return out
    
    @app.callback(Output('figure_violin_features', 'figure'),
                 [Input('dropdown_violin_features', 'value')])
    def _update_violin_plot_features(values):
        df = X[values] # TODO also only use important features
        out = _get_violin_distribution(df)
        return out
    
    @app.callback(Output('figure_class_based_violin_features', 'figure'),
                 [Input('dropdown_class_based_violin_features', 'value')])
    def _update_class_based_violin_plot_features(value):
        # TODO also only use important features?
        out = _get_class_based_violin_distribution(X, Y, value, LE_NAME_MAPPING, OUTPUT_PATH)
        return out
    
    @app.callback(Output('figure_violin_custom_features', 'figure'),
                 [Input('dropdown_violin_custom_features', 'value')])
    def _update_violin_plot_custom_features(values):
        df = X[values]
        out = _get_violin_distribution(df)
        return out
    
    @app.callback(Output('figure_basic_model_comparison', 'figure'),
                 [Input('dropdown_basic_model_comparison', 'value')])
    def _update_basic_model_comparison(values):
        selected_model_comparison = MODEL_COMPARISON.loc[MODEL_COMPARISON['value'].isin(values)]
        fig_comparison = compute_fig_from_df(MODEL_TYPE, selected_model_comparison, METRICS)
        
        return fig_comparison
    
    @app.callback(Output('figure_basic_model_scatter_plot', 'figure'),
                  [Input('dropdown_basic_model_scatter_plot', 'value')])
    def _update_basic_model_scatter_plot():
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        regressor = DummyRegressor()
        out = cross_validate(regressor, X_train, Y_train, scoring = ['neg_mean_absolute_error'], return_estimator= True)
        pred_dummy = out['estimator'][0].predict(X_test)

        regressor = get_model_with_name_regression(value[0])
        out = cross_validate(regressor, X_train, Y_train, scoring = ['neg_mean_absolute_error'], return_estimator= True)
        pred_rf = out['estimator'][0].predict(X_test)
        
        # convert prediction to pandas 
        tmp_df = pd.DataFrame(np.array([Y_test, pred_rf]).transpose(), columns= ['baseline', 'prediction'])
        tmp_df['model'] = 'random forest'

        tmp_df2 = pd.DataFrame(np.array([Y_test, pred_dummy]).transpose(), columns= ['baseline', 'prediction'])
        tmp_df2['model'] = 'dummy'

        tmp = pd.concat([tmp_df, tmp_df2])
        
        fig = px.scatter(tmp, x='prediction', y='baseline', color= 'model', marginal_x="histogram", marginal_y="histogram")
        
        return fig

######################### Helper #########################

def add_dataframe_table(df: pd.DataFrame, id=None, className='table'):
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
        ], className=className)
    else:
        out = html.Div([        
            dash.dash_table.DataTable(df.to_dict('records'),[{'name': i, 'id': i} for i in df.columns]),
        ], className=className)
    return out
