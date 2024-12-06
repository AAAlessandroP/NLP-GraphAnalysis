import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle
from dash import callback
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter
import pickle

app = dash.Dash(__name__)

with open('flat_list.pickle', 'rb') as f:
    flat_list = pickle.load(f)
c=Counter(flat_list)#per la wordcloud

import pandas as pd
from datetime import datetime, time as tm
df = pd.read_csv('df.csv')
listaUtentiRilevanti = ["Uncle_SamCoco", "RamiAlLolah", "warrnews", "WarReporter1", "mobi_ayubi", "_IshfaqAhmad"]

app.layout = html.Div([

    html.H1("Visualizing Frequent words in the data", style={'float':'left', 'textAlign': 'center', 'color': '#333', 'font-family': 'Arial'}),    
        html.Div(id='img-in-here', children='wordcloud placeholder',style={
            'display': 'flex',
            'justifyContent': 'space-around',
            'align-items': 'stretch',
            'backgroundColor': '#f9f9f9',
            'padding': '5px',
            'margin': '5px auto',
            'border': '1px solid black',
            'borderRadius': '5px',
            'width': '100%',
            'textAlign': 'center',
            'fontSize': '14px',
            'lineHeight': '1.2'
    }), 

    html.Div([
        html.Label('Adjust minimum word frequence', style={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '14px'}),
        dcc.Slider(
            id='word-threshold', min=0, max=1500, step=100, value=0, 
            marks={i: str(i) for i in [0, 0.5, 1]},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
    ], style={'width': '100%', 'padding': '0px 10px', 'box-sizing': 'border-box'}),



    html.Div([
        html.H1("Sentiment over time", style={'float':'center', 'textAlign': 'center', 'color': '#333', 'font-family': 'Arial'}),
    ], style={'width': '100%', 'padding': '0px 10px', 'box-sizing': 'border-box'}),

    html.Div(id='time-in-here', children='timeplot placeholder',style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'align-items': 'stretch',
        'backgroundColor': '#f9f9f9',
        'padding': '5px',
        'margin': '5px auto',
        'border': '1px solid black',
        'borderRadius': '5px',
        'width': '100%',
        'textAlign': 'center',
        'fontSize': '14px',
        'lineHeight': '1.2'
    }), 

    html.Div([
        html.Label('Select user/users to display: ', style={'float':'center', 'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '14px'}),
        dcc.Checklist(
            listaUtentiRilevanti,
            id="checkbox",
            #inline=True#orizzontali,
            value=["warrnews"],
            style={'float':'center', 'textAlign': 'center', 'color': '#333', 'font-family': 'Arial'}
        ),
    ]),
])

@callback(
    Output('img-in-here', 'children'),
    Input('word-threshold', 'value'),    
    prevent_initial_call=False
)
def update_output(value):
    soglia=value
    cc={}
    for w in c:
        if c[w]>=soglia:
            cc[w] = c[w]
    wordcloud = WordCloud(width=1600, height=800, background_color="white").generate_from_frequencies(cc)

    img = wordcloud
    fig = px.imshow(img)
    return dcc.Graph(id="graph-picture", figure=fig)

@callback(
    Output('time-in-here', 'children'),
    Input('checkbox', 'value'),    
    prevent_initial_call=False
)
def update_output_time(utentiSelezionati):
    datiX=[]
    datiY=[]
    if utentiSelezionati is None or len(utentiSelezionati) == 0:
        return
    
    for user in utentiSelezionati:
        suoi = df[df["username"]==user]
        datiX.append(suoi["time"].apply(datetime.strptime, args=("%m/%d/%Y %H:%M",)))
        datiY.append(suoi["tweetSentimentWithoutPreProc"].apply(lambda x: x/5))

    mydata = []
    for i in range(len(utentiSelezionati)):
        user = utentiSelezionati[i]
        # print("user, i: ", user, i)
        mydata.append({'x': datiX[i], 'y': datiY[i], 'type': 'bar', 'name': user,
                       'transforms':[dict(
                            type = 'aggregate',                            
                            aggregations = [dict(
                                target = 'y', func = 'avg', enabled = True)#sennò somma i sentiment dei suoi tweet di quel giorno!
                            ]
                        )]
                       })
        
    # fig2 = go.Figure(data=[go.Scatter(x=suoi["time"], y=suoi["tweetSentimentWithoutPreProc"])])

    return dcc.Graph(
    figure={
        'data': mydata,
        'layout': {
            'title': 'Day-long averaged Sentiment of user\'s tweets',
                'xaxis':{
                    'title':'Day of the tweet'
                },
                'yaxis':{
                     'title':'Average sentiment '
                }
        }
    })

if __name__ == '__main__':
    app.run_server(debug=True)