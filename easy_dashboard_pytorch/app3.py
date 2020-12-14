import base64
import datetime
import io
import sys
import dash
#import urllib
import six.moves.urllib as urllib
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import pandas as pd
import zipfile
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
f = open('out.txt', 'w')
f.close()
app = dash.Dash(__name__)
#server = app.server
df = []
app.layout = html.Div([
                html.Div([
                    html.H1("Python Dash ANN Dashboard",style={"textAlign": "center"}),
                    #html.Img(src="/assets/CollegeofEngineeringLogog.png"),
                    html.Img(src="/assets/UGALogo.jpg")
                ],className="banner"),
                dcc.Tabs(id="tabs", children=[
                    dcc.Tab(label='ANN MODELING', children=[
                        #html.H6("Remove N/A"),
                        #html.Div(id='model-accuracy'),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div([
                                html.H6(""),
                                html.H6(""),
                                html.H6(""),
                                #html.H6("Drop feature:"),
                                dcc.Input(id="input4", type="Text", placeholder="Input Nodes :"),
                                html.P(),
                                dcc.Input(id="input_1", type="Text", placeholder="Output Nodes :"),
                                html.P(),
                                #html.H6("Scaler:"),
                                #html.Div([
                                #    dcc.Dropdown(
                                #        id='select-scaler',
                                #        options=[
                                #            {'label': 'Standard Scaler', 'value': 'SS'},
                                #            {'label': 'Min Max Scalar', 'value': 'SMMS'},
                                #            {'label': 'Max Abs Scaler', 'value': 'MAS'},
                                #            {'label': 'Normalizer', 'value': 'NORM'}
                                #            ],
                                        #value='SS',
                                #        placeholder='Select Scalar...',
                                #        style={
                                #            'width':'50%'
                                #        }
                                #    )
                                #]),
                                #html.H6("Number of layer"),
                                dcc.Input(id="input5", type="number", placeholder="Number of Hidden Layer"),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        html.Button('Upload File')
                                    ],style={
                                        'textAlign': 'left',
                                    }),
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                                #html.P(),
                                #html.I("Type Activation Function From Below List:"),
                                #html.P(),
                                #html.I("1.>Sigmoid activation function 2.> Linear activation function 3.>RELU 4.>ELU"),
                                #html.P(),
                                #dcc.Input(id="input6", type="text", placeholder="Activation Function"),
                                #html.H6("Select activation function:"),
                                html.Div([
                                    #html.H6("Enter Test, Train and Validation Ratio:"),
                                    dcc.Input(id="input1", type="number", placeholder="Train %"),
                                    dcc.Input(id="input2", type="number", placeholder="Test %"),
                                    dcc.Input(id="input3", type="number", placeholder="Validation %"),
                                    ]
                                ),
                                html.Div([
                                    dcc.Dropdown(
                                        id='select-activation-function',
                                        options=[
                                            {'label': 'Sigmoid activation function', 'value': 'sigmoid'},
                                            {'label': 'Linear activation function', 'value': 'linear'},
                                            {'label': 'Rectified linear unit activation function', 'value': 'relu'},
                                            {'label': 'Exponential linear unit', 'value': 'elu'}
                                            ],
                                        #value='sigmoid',
                                        placeholder='Select activation function.....',
                                        style={
                                            'width':'50%'
                                        }
                                    )
                                ]),
                                #html.H6(""),
                                #html.H6("Select Optimizer"),
                                html.Div([
                                    dcc.Dropdown(
                                        id='select-optimizer',
                                        options=[
                                            #{'label':'Adadelta algorithm', 'value':'Adadelta'},
                                            #{'label':'Adagrad algorithm', 'value':'Adagrad'},
                                            {'label':'ADAM Optimizer','value':'Adam'},
                                            {'label':'RMSprop algorithm','value':'RMSprop'},
                                            {'label': 'Stochastic Gradient Descent ', 'value': 'SGD'},
                                            #{'label': 'Stochastic Gradient Descent with Momentum', 'value': 'SGDM'},
                                            {'label': 'Particle Swarm Optimization', 'value': 'PSO'},
                                            {'label': 'Gradient Particle Swarm Optimization', 'value': 'GPSO'}
                                            ],
                                        #value='SGD',
                                        placeholder='Select Optimizer...',
                                        style={
                                            'width':'50%'
                                        }
                                    )
                                ]),
                                html.P(),
                                html.Button(id='submit-button-1',
                                            n_clicks=0,
                                            children='Run',
                                            style={'fontSize':14}
                                ),
                                html.P(),
                                html.Button(
                                    html.A('Download',
                                        id='download-link',
                                        download="predictions.csv",
                                        href="",
                                        target="_blank"
                                    )
                                ),
                                html.P(),
                                #html.H6("The accuracy obtained for the model:"),
                                html.Div(id='model-create')
                                ],id='model-training-1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'background-color':'white',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                }),
                                html.Div([
                                    html.Iframe(id='console-out',srcDoc='',style={'width':'100%','background-color': 'rgb(250,250,250)','height':650})
                                ],id='model-training-2',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'background-color':'white',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })

                            ])
                        ],className="inside_block"),
                        html.Div(id='model-prediction',className="border"),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div(id='block-label',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'background-color':'white',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                }),
                                html.Div(id='block-label1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'background-color':'white',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })
                        ])],className="border"),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div(id='ann-graph',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'background-color':'white',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                }),
                                html.Div(id='ann-graph1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'background-color':'white',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })
                        ])],className="border")]),
                dcc.Tab(label='IANN MODELING', children=[html.Div([
                    html.Div(className='row',children=[
                        html.Div([
                        html.H6(""),
                        html.H6(""),
                        html.H6(""),
                        html.Div([
                            #html.H6("Enter Test, Train and Validation Ratio:"),
                            dcc.Input(id="input6", type="number", placeholder="Train %"),
                            dcc.Input(id="input7", type="number", placeholder="Test %"),
                            dcc.Input(id="input8", type="number", placeholder="Validation %"),
                            ]
                        ),
                        dcc.Input(id="input9", type="Text", placeholder="Drop feature"),
                        dcc.Input(id="input10", type="number", placeholder="Number of Layer"),
                        html.Div([
                            dcc.Dropdown(
                                id='select-activation-function-1',
                                options=[
                                    {'label': 'Sigmoid activation function', 'value': 'sigmoid'},
                                    {'label': 'Linear activation function', 'value': 'linear'},
                                    {'label': 'Rectified linear unit activation function', 'value': 'relu'},
                                    {'label': 'Exponential linear unit', 'value': 'elu'}
                                    ],
                                #value='sigmoid',
                                placeholder='Select activation function.....',
                                style={
                                    'width':'50%'
                                }
                            )
                        ]),
                        html.Div([
                            dcc.Dropdown(
                                id='select-optimizer-1',
                                options=[
                                    {'label': 'Stochastic Gradient Descent ', 'value': 'SGD'},
                                    {'label': 'Stochastic Gradient Descent with Momentum', 'value': 'SGDM'},
                                    {'label': 'Particle Swarm Optimization', 'value': 'PSO'},
                                    {'label': 'Gradient Particle Swarm Optimization', 'value': 'GPSO'}
                                    ],
                                placeholder='Select Optimizer...',
                                style={
                                    'width':'50%'
                                }
                            )
                        ]),
                        html.P(),
                        html.Button(id='submit-button-2',
                                    n_clicks=0,
                                    children='Submit',
                                    style={'fontSize':14}),
                        html.P(),
                        html.H6("The accuracy obtained for the model:"),
                        html.Div(id='model-create-1')
                        ],id='model-training-3',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'background-color':'white',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        }),
                        html.Div(id='model-training-4',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'background-color':'white',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        })

                    ])
                ],className="border"),
                html.Div([
                    html.Div(className='row',children=[
                        html.Div(id='block-label2',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'borderRadius': '5px',
                            'background-color':'white',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        }),
                        html.Div(id='block-label3',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'background-color':'white',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        })
                ])],className="border"),
                html.Div([
                    html.Div(className='row',children=[
                        html.Div(id='iann-graph',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'borderRadius': '5px',
                            'background-color':'white',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        }),
                        html.Div(id='iann-graph1',
                        className="five column",
                        style={
                            'width': '48%',
                            'height': '680px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'border':'2px black solid',
                            'background-color':'white',
                            'borderRadius': '5px',
                            'textAlign': 'left',
                            'margin': '10px',
                            'alignSelf': 'stretch',
                            'display': 'inline-block'
                        })
                ])],className="border")
                    ]),
                    dcc.Tab(label='TRAIN IMAGES', children=[
                        #html.H6("Remove N/A"),
                        #html.Div(id='model-accuracy'),
                        html.Div([
                            html.Div(className='row',children=[
                                html.Div([
                                html.H6(""),
                                html.H6(""),
                                html.H6(""),
                                dcc.Upload(
                                    id='upload-train-data',
                                    children=html.Div([
                                        html.Button('Upload Training Data in zip format')
                                    ],style={
                                        'textAlign': 'left',
                                    }),
                                    accept=".zip",
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                                dcc.Upload(
                                    id='upload-test-data',
                                    children=html.Div([
                                        html.Button('Upload Testing Data in zip format')
                                    ],style={
                                        'textAlign': 'left',
                                    }),
                                    accept=".zip",
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                                dcc.Upload(
                                    id='upload-validation-data',
                                    children=html.Div([
                                        html.Button('Upload Validation Data in zip format')
                                    ],style={
                                        'textAlign': 'left',
                                    }),
                                    accept=".zip",
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                ),
                                html.Div([
                                    #html.H6("Enter Test, Train and Validation Ratio:"),
                                    dcc.Input(id="input_image1", type="number", placeholder="Number of Filter... "),
                                    dcc.Input(id="input_image2", type="number", placeholder="Pooling..."),
                                    dcc.Input(id="input_image3", type="number", placeholder="Strides..."),
                                    #dcc.Input(id="input_image4", type="number", placeholder="Input Shape..."),
                                    ]
                                ),
                                dcc.Input(id="input_image5", type="number", placeholder="Number of Convolutional Layer..."),
                                dcc.Input(id="input_image6", type="number", placeholder="Number of Hidden Layer..."),
                                html.Div([
                                    dcc.Dropdown(
                                        id='select-optimizer-image',
                                        options=[
                                            #{'label':'Adadelta algorithm', 'value':'Adadelta'},
                                            #{'label':'Adagrad algorithm', 'value':'Adagrad'},
                                            {'label':'ADAM Optimizer','value':'Adam'},
                                            {'label':'RMSprop algorithm','value':'RMSprop'},
                                            {'label': 'Stochastic Gradient Descent ', 'value': 'SGD'},
                                            #{'label': 'Stochastic Gradient Descent with Momentum', 'value': 'SGDM'},
                                            {'label': 'Particle Swarm Optimization', 'value': 'PSO'},
                                            {'label': 'Gradient Particle Swarm Optimization', 'value': 'GPSO'}
                                            ],
                                        #value='SGD',
                                        placeholder='Select Optimizer...',
                                        style={
                                            'width':'50%'
                                        }
                                    )
                                ]),
                                html.P(),
                                html.Button(id='submit-button-3',
                                            n_clicks=0,
                                            children='Run',
                                            style={'fontSize':14}
                                ),
                                html.P(),
                                html.Button(
                                    html.A('Download',
                                        id='download-model-link',
                                        download="model.h5",
                                        href="",
                                        target="_blank"
                                    )
                                ),
                                html.P(),
                                #html.H6("The accuracy obtained for the model:"),
                                html.Div(id='model-image-create')
                                ],id='model-image-training-1',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'background-color':'white',
                                    'border':'2px black solid',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',


                                    'display': 'inline-block'
                                }),
                                html.Div([
                                    html.Iframe(id='console-out-image',srcDoc='',style={'width':'100%','background-color': 'rgb(250,250,250)','height':650})
                                ],id='model-image-training-2',
                                className="five column",
                                style={
                                    'width': '48%',
                                    'height': '680px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'border':'2px black solid',
                                    'background-color':'white',
                                    'borderRadius': '5px',
                                    'textAlign': 'left',
                                    'margin': '10px',
                                    'alignSelf': 'stretch',
                                    'display': 'inline-block'
                                })

                            ])
                        ],className="inside_block"),
                        html.Div(id='model-prediction-image',className="border"),
                        html.Div(
                            html.Div([
                                dcc.Upload(
                                    id='upload-image',
                                    children=html.Div([
                                        html.Button('Upload Image to Test')
                                    ],style={
                                        'textAlign': 'left',
                                    })
                                ),
                                html.Hr(),
                                html.Div(id='predicted-image')
                            ],
                            style={
                                'width': '88%',
                                'height': '680px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'border':'2px black solid',
                                'background-color':'white',
                                'borderRadius': '5px',
                                'textAlign': 'left',
                                'margin': '10px',
                                'alignSelf': 'stretch',
                                'display': 'inline-block'
                            }),className="border"
                        )
                        ])
                ])
],className="dashboard")

def datasetData(contents,filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

def datasetDataWithoutException(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return df


def labelName(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        html.H6('Select Column Name:'),
        dcc.Checklist(
            id='labelCheckList',
            options=[{'label': i, 'value': i} for i in df.columns],
            value=[df.columns[0]],
            labelStyle={'display': 'inlineBlock', 'marginRight': '40px'},
			inputStyle={'marginRight': '7.5px'},
            #style={
            #    'textAlign': 'left',
            #}
        )
])

def data_table(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        html.H6("Data Table of Selected Data:"),
        html.Div(id = "dash-table")
])

def annGraph1(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        dcc.Dropdown(
        id='graph-input1',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[0]
        ),
        dcc.Dropdown(
        id='graph-input2',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[1]
        ),
        html.Div(id = 'graph1')
])


def annGraph2(contents, filename, date):
    df = datasetData(contents, filename)
    return html.Div([
        dcc.Dropdown(
        id='graph-input3',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[0]
        ),
        dcc.Dropdown(
        id='graph-input4',
        options=[
            {'label': i, 'value': i} for i in df.columns
        ],
        value= df.columns[1]
        ),
        html.Div(id = 'graph2')
])


def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)
    return im

@app.callback(Output('block-label', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def labelCheckList(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            labelName(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('block-label1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def dashDataTable(list_of_contents, list_of_names, list_of_dates):
    #print(list_of_contents, list_of_names, list_of_dates)
    if list_of_contents is not None:
        #print("Akash Saurabh")
        #print(zip(list_of_contents, list_of_names, list_of_dates))
        children = [
            data_table(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('dash-table', 'children'),
              [Input('labelCheckList', 'value'),
               Input('upload-data', 'contents')])
def dataTable(labelName, list_of_contents):
    content_type, content_string = list_of_contents[0].split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df=df[labelName]
    df=df.head(8)
    return html.Div([
        #html.H6("To Test the function work"),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        )
])

@app.callback(Output('ann-graph', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def graphPart1(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            annGraph1(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('graph1', 'children'),
              [Input('upload-data', 'contents'),
               Input('graph-input1','value'),
               Input('graph-input2','value')])
def Graph1(list_of_contents1, value1, value2):
    content_type, content_string = list_of_contents1[0].split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return html.Div(dcc.Graph(
        figure={'data':[
            go.Scatter(
                x=df[value1],
                y=df[value2],
                dy=1,
                mode='markers',
                marker={'size':15}
    )],'layout':go.Layout(title='First Figure',hovermode='closest')
    }))


@app.callback(Output('ann-graph1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def graphPart2(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            annGraph2(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('graph2', 'children'),
              [Input('upload-data', 'contents'),
               Input('graph-input3','value'),
               Input('graph-input4','value')])
def Graph1(list_of_contents1, value1, value2):
    content_type, content_string = list_of_contents1[0].split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return html.Div(dcc.Graph(
        figure={'data':[
            go.Scatter(
                x=df[value1],
                y=df[value2],
                dy=1,
                mode='lines',
                marker={'size':15}
    )],'layout':go.Layout(title='Second Figure',hovermode='closest')
    }))

#@app.callback(Output('model-accuracy', 'children'),
#              [Input('upload-data', 'contents'),
#               Input('input1','value'),
#               Input('input2','value'),
#               Input('input3','value'),
#               Input('input4','value'),
#               Input('input5','value'),
#               Input('input6','value'),
#               Input('select-optimizer','value')])
#def createModel1(list_of_contents2, value1, value2, value3, value4, value5, value6, value7):
    #print(value1, value2, value3, value4)
#    content_type, content_string = list_of_contents2[0].split(',')
#    decoded = base64.b64decode(content_string)
#    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#    return html.Div(df)

def accuracyCal(true,pred):
    acc = (true.argmax(-1) == pred.argmax(-1))
    return (100 * acc.sum() / len(acc))

@app.callback(Output('model-prediction', 'children'),
              [Input('upload-data','contents'),
               Input('submit-button-1', 'n_clicks')],
              [State('input1','value'),
               State('input2','value'),
               State('input3','value'),
               State('input4','value'),
               State('input_1','value'),
               State('input5','value'),
               #State('select-scalar','value'),
               #State('input6','value'),
               State('select-activation-function','value'),
               State('select-optimizer','value')])
def createModel1_1(list_of_contents2,num_clicks, train, test, val, inputNode, outputNode, numLayer, actFunc, optimizer):
    #print(value1, value2, value3, value4)
    print(train, test, val, inputNode, outputNode, numLayer, actFunc, optimizer)
    print(list_of_contents2)
    content_type, content_string = list_of_contents2[0].split(',')
    #print(content_type)
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    #print(df)
    #print(df[0])
    #print(outputNode)
    inputNode = inputNode.split(',')
    outputNode = outputNode.split(',')
    #print(outputNode)
    #print(label[2])
    #df = df.drop(label,axis=1)
    #X = df.drop(df.columns[0],axis=1)
    X = df[inputNode]
    #y = df[df.columns[0]]
    y = df[outputNode]
    #print(X.shape)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dropout
    from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
    import torch
    from torch import nn
    import torch.nn.functional as F

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test,random_state=101)
    #print("Shape of training data:")
    #print(X_train.shape)
    #print(y.shape[1])
    scaler = MinMaxScaler()
    X_train= scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # model = Sequential()
    # #print(numLayer)
    # for x in range(numLayer):
    #     #print(x)
    #     model.add(Dense(X_train.shape[1],activation=actFunc))
    #     #model.add(Dropout(0.5))
    # #model.add(Dense(X_train.shape[1],activation='relu'))
    # #model.add(Dense(X_train.shape[1],activation='relu'))
    # #model.add(Dense(X_train.shape[1],activation='relu'))
    # model.add(Dense(y.shape[1]))
    # model.compile(optimizer=optimizer,loss='mse')
    # early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    n_input_dim = X_train.shape[1]
    # Layer size
    n_hidden = X_train.shape[1] # Number of hidden nodes
    n_output = y_train.shape[1] # Number of output nodes for predicted mpg

    # Build mdel
    torch_model = torch.nn.Sequential(
    torch.nn.Linear(n_input_dim, n_hidden),
    torch.nn.ELU(),
    torch.nn.Linear(n_input_dim, n_hidden),
    torch.nn.ELU(),
    torch.nn.Linear(n_input_dim, n_hidden),
    torch.nn.ELU(),
    torch.nn.Linear(n_input_dim, n_hidden),
    torch.nn.ELU(),
    torch.nn.Linear(n_input_dim, n_hidden),
    torch.nn.ELU(),
    torch.nn.Linear(n_input_dim, n_hidden),
    torch.nn.ELU(),
    torch.nn.Linear(n_hidden, n_output))
    train_error = []
    iters = 80000

    Y_train_t = torch.FloatTensor(y_train.values)#Converting numpy array to torch tensor
    X_train_t = torch.FloatTensor(X_train)  #Converting numpy array to torch tensor
    loss_func = torch.nn.MSELoss() #Choosing mean square error as loss metric
    learning_rate = 0.01
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=learning_rate)

    #print('Akash')

    orig_stdout = sys.stdout
    f = open('out.txt', 'a')
    sys.stdout = f
    # model.fit(x=X_train,y=y_train.values,
    #       validation_data=(X_test,y_test.values),
    #       batch_size=128,epochs=400,
    #       verbose=1,
    #       callbacks=[early_stop])
    #print("Model creation complete...")
    for i in range(iters):
        y_hat = torch_model(X_train_t)
        loss = loss_func(y_hat, Y_train_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_error.append(loss.item())

    #print('train.error',train_error)
    sys.stdout = orig_stdout
    f.close()
    #losses = pd.DataFrame(model.history.history)
    #predict = model.predict(X_test)
    X_test_t = torch.FloatTensor(X_test)
    predict = torch_model(X_test_t)
    predictions = []
    for elem in predict:
        predictions.extend(elem)
    pred_df = pd.Series(predictions, name='predictions')
    y_test = y_test.reset_index(drop = True)
    result = pd.concat([y_test, pred_df], axis=1)
    #predictions.insert(0, "predictions")
    #predictions = pd.DataFrame(predictions)
    #print("Prediction1:",predictions)
    #print("Prediction:",y_test)
    mean_absolute_error = mean_absolute_error(y_test,predict.detach().numpy())
    mean_square_erro = mean_squared_error(y_test,predict.detach().numpy(),squared=True)
    accuracy = accuracyCal(y_test.to_numpy(),predict.detach().numpy())
    #pred_df = [predictions,y_test]
    #result = pd.concat(pred_df)
    #print(result)
    #print("Test01")
    #pred_df = pd.DataFrame (pred_df)
    #print("Test02")
    #print(pred_df)

    result.to_csv('predictions.csv')

    #print(losses)
    #print(predictions)
    #print("AKASH")
    #print(pred_df)
    #print(pred_df['price'])

    #print(predictions.shape, y_test.shape)
    loss = go.Scatter(x=train_error,mode='lines')
    #val_loss = go.Scatter(x=train_error,mode='lines')
    data1 = [loss]#,val_loss]
    test_data = go.Scatter(x=result[result.columns[0]], y=result[result.columns[1]], mode='markers')
    real_data = go.Scatter(x=result[result.columns[0]], y=result[result.columns[0]], mode='lines')
    data2 = [test_data, real_data]
    return html.Div([
        dcc.Interval(id='interval1', interval=1 * 1000, disabled=True, n_intervals=0),
        dcc.Interval(id='interval2', interval=5 * 1000, disabled=True, n_intervals=0),
        html.Div([

            html.Div(className='row',children=[
                html.Div([
                    #html.H2("Performance Metrics Regression Prediction", style={"textAlign": "left"}),
                    html.H6("The accuracy in the Training set equal to :"),str(accuracy),
                    html.H6("Mean Absolute Error : "), str(mean_absolute_error),
                    dcc.Graph(figure={'data':data1,'layout':go.Layout(title='Loss Vs Validation Loss ',hovermode='closest')})
                ],id='val_loss_graph',
                className="five column",
                style={
                    'width': '48%',
                    'height': '680px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'border':'2px black solid',
                    'borderRadius': '5px',
                    'background-color':'white',
                    'textAlign': 'left',
                    'margin': '10px',
                    'alignSelf': 'stretch',
                    'display': 'inline-block'
                }),
                html.Div([
                    html.H6("Mean Square Error : "),str(mean_square_erro),
                    html.H6("trace0 represents graph between Test data and Predicted data"),
                    html.H6("trace1 represents graph of Test data vs Test data"),
                    dcc.Graph(figure={'data':data2,'layout':go.Layout(title='Test data vs Predicted data ',hovermode='closest')})
                ],id='test_real_graph',
                className="five column",
                style={
                    'width': '48%',
                    'height': '680px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'border':'2px black solid',
                    'background-color':'white',
                    'borderRadius': '5px',
                    'textAlign': 'left',
                    'margin': '10px',
                    'alignSelf': 'stretch',
                    'display': 'inline-block'
                })
        ])])

    ])

@app.callback(Output('div-out','children'),
             [Input('interval1', 'n_intervals')])
def update_interval(n):
    #orig_stdout = sys.stdout
    #f = open('out.txt', 'a')
    #sys.stdout = f
    print ('Intervals Passed: Akash ' + str(n))
    #sys.stdout = orig_stdout
    #f.close()
    return 'Intervals Passed: ' + str(n)

@app.callback(Output('console-out','srcDoc'),
             [Input('interval2', 'n_intervals')])
def update_output(n):
    file = open('out.txt', 'r')
    data=''
    lines = file.readlines()
    if lines.__len__()<=20:
        last_lines=lines
    else:
        last_lines = lines[-20:]
    for line in last_lines:
        data=data+line + '<BR>'
    file.close()
    print(data)
    return data

@app.callback(Output('console-out-image','srcDoc'),
             [Input('interval2', 'n_intervals')])
def update_output(n):
    file = open('out.txt', 'r')
    data=''
    lines = file.readlines()
    if lines.__len__()<=20:
        last_lines=lines
    else:
        last_lines = lines[-20:]
    for line in last_lines:
        data=data+line + '<BR>'
    file.close()
    print(data)
    return data

@app.callback(Output('download-link','href'),
             [Input('submit-button-1','n_clicks')])
def downloadCSV(num_click):
    dff = pd.read_csv('./predictions.csv')
    csv_string = dff.to_csv(index=False, encoding='utf-8')
    #print("saurabh")
    #print(csv_string)
    #print(urllib.parse.quote(csv_string))
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


# @app.callback(Output('predicted-image-return', 'children'),
#               [Input('upload-image','contents')])
# def predictImage(image):
#     print(image)
#     return html.Div([
#         html.Img(src=contents),
#         html.Hr(),
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])



@app.callback(Output('predicted-image', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def image_prediction(predict_image, list_of_names, list_of_dates):
    print("AKSHS",type(predict_image), list_of_names, list_of_dates)
    #string = predict_image.split(';base64,')[-1]
    #im_pil = b64_to_pil(string)
    #print(type(im_pil),im_pil)
    data = predict_image.encode("utf8").split(b";base64,")[1]
    #predict_image_type,predict_image_string = predict_image.split(',')
    #predict_image_decoded = base64.b64decode(predict_image_string)
    #predict_image_str = io.BytesIO(predict_image_decoded)
    import numpy as np
    from keras.preprocessing import image
    from tensorflow.keras.models import load_model
    cnn = load_model('model.h5')
    fh = open("imageToPredict.jpeg", "wb")
    fh.write(base64.decodebytes(data))
    fh.close()
    test_image = image.load_img('./imageToPredict.jpeg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    #training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    return html.Div([
        #html.H5(list_of_names),
        #html.H6(datetime.datetime.fromtimestamp(list_of_dates)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=predict_image,style={
            'width': '256px',
            'height': '256px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'border':'2px black solid',
            'borderRadius': '5px',
            'background-color':'white',
            'textAlign': 'left',
            'margin': '10px',
            'alignSelf': 'stretch',
            'display': 'inline-block'
        }),
        html.H6("The Uploaded Image is :"),
        html.H6(prediction),
        #html.Div('Raw Content'),
        # html.Pre(list_of_contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])



@app.callback(Output('model-prediction-image', 'children'),
              [Input('upload-train-data','contents'),
               Input('upload-test-data','contents'),
               Input('upload-validation-data','contents'),
               Input('submit-button-3', 'n_clicks')],
              [State('input_image1','value'),
               State('input_image2','value'),
               State('input_image3','value'),
               #State('input_image4','value'),
               State('input_image5','value'),
               State('input_image6','value'),
               State('select-optimizer-image','value')])
def createModelImage(trainImage,testImage, validationImage, num_click, numFilter, pooling, strides, numConvLayer, numLayer, optimizer):
    print( num_click, numFilter, pooling, strides,  numConvLayer, numLayer, optimizer)
    trainImage = " ".join(str(x) for x in trainImage)
    #print("Saurabh")
    #print(trainImage)
    # try:
    #     train_type,train_string = trainImage.split(',')
    #     #print("Akash")
    #     #print(train_type,train_string)
    #     train_decoded = base64.b64decode(train_string)
    #     train_str = io.BytesIO(train_decoded)
    #     train_obj = zipfile.ZipFile(train_str, 'r')
    #     train_obj.extractall('./Dataset/')
    # except :
    #     return html.Div([
    #         dcc.ConfirmDialog(
    #             message='Please upload training dataset',
    #             )
    #     ])
    #
    # try:
    #     testImage = " ".join(str(x) for x in testImage)
    #     test_type, test_string = testImage.split(',')
    #     test_decoded = base64.b64decode(test_string)
    #     test_str = io.BytesIO(test_decoded)
    #     test_obj = zipfile.ZipFile(test_str, 'r')
    #     test_obj.extractall('./Dataset/')
    #     print(test_obj)
    # except :
    #     return html.Div([
    #         dcc.ConfirmDialog(
    #             message='Please upload training dataset',
    #             )
    #     ])
    # try:
    #     validationImage = " ".join(str(x) for x in validationImage)
    #     validation_type, validation_string = validationImage.split(',')
    #     validation_decoded = base64.b64decode(validation_string)
    #     validation_str = io.BytesIO(validation_decoded)
    #     validation_obj = zipfile.ZipFile(validation_str, 'r')
    #     validation_obj.extractall('./Dataset/')
    # except :
    #     print("Validation Data Not uploaded")
    #     #return html.Div([
    #     #    dcc.ConfirmDialog(
    #     #        message='Please upload training dataset',
    #     #        )
    #     #])
    #
    # #### Preprocessing the Training set
    # train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    # training_set = train_datagen.flow_from_directory('./Dataset/training_set', target_size = (64, 64), batch_size = 8, class_mode = 'binary')
    # ### Preprocessing the Test set
    # test_datagen = ImageDataGenerator(rescale = 1./255)
    # test_set = test_datagen.flow_from_directory('./Dataset/test_set', target_size = (64, 64), batch_size = 8, class_mode = 'binary')
    # ### Initialising the CNN
    # cnn = tf.keras.models.Sequential()
    # for x in range(numConvLayer):
    #     cnn.add(tf.keras.layers.Conv2D(filters=numFilter, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    #     cnn.add(tf.keras.layers.MaxPool2D(pool_size=pooling, strides=strides))
    # cnn.add(tf.keras.layers.Flatten())
    # for x in range(numLayer):
    #     cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # cnn.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    # orig_stdout = sys.stdout
    # f = open('out.txt', 'a')
    # sys.stdout = f
    # cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
    # sys.stdout = orig_stdout
    # f.close()
    # cnn.save('model.h5')
    # losses = pd.DataFrame(cnn.history.history)
    #
    # loss = go.Scatter(x=losses['loss'],mode='lines')
    # val_loss = go.Scatter(x=losses['val_loss'],mode='lines')
    # data1 = [loss,val_loss]
    # accuracy = go.Scatter(x=losses['accuracy'], mode='lines')
    # val_acc = go.Scatter(x=losses['val_accuracy'], mode='lines')
    # data2 = [accuracy, val_acc]
    # return html.Div([
    #     dcc.Interval(id='interval1', interval=1 * 1000, disabled=True, n_intervals=0),
    #     dcc.Interval(id='interval2', interval=5 * 1000, disabled=True, n_intervals=0),
    #     html.Div([
    #         html.Div(className='row',children=[
    #             html.Div([
    #                 dcc.Graph(figure={'data':data1,'layout':go.Layout(title='Loss Vs Validation Loss ',hovermode='closest')})
    #             ],id='val_loss_image_graph',
    #             className="five column",
    #             style={
    #                 'width': '48%',
    #                 'height': '680px',
    #                 'lineHeight': '60px',
    #                 'borderWidth': '1px',
    #                 'border':'2px black solid',
    #                 'borderRadius': '5px',
    #                 'background-color':'white',
    #                 'textAlign': 'left',
    #                 'margin': '10px',
    #                 'alignSelf': 'stretch',
    #                 'display': 'inline-block'
    #             }),
    #             html.Div([
    #                 dcc.Graph(figure={'data':data2,'layout':go.Layout(title='Accuracy vs Val_Accuracy ',hovermode='closest')})
    #             ],id='acc_val_acc_image_graph',
    #             className="five column",
    #             style={
    #                 'width': '48%',
    #                 'height': '680px',
    #                 'lineHeight': '60px',
    #                 'borderWidth': '1px',
    #                 'border':'2px black solid',
    #                 'background-color':'white',
    #                 'borderRadius': '5px',
    #                 'textAlign': 'left',
    #                 'margin': '10px',
    #                 'alignSelf': 'stretch',
    #                 'display': 'inline-block'
    #             })
    #     ])])
    #
    # ])

app.css.append_css({
    "external_url":"https://codepen.io/chriddyp/pen/bWLwgP.css"
})


if __name__ == '__main__':
    app.run_server()
