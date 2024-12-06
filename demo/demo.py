import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import igraph as ig
import pickle
import numpy as np
from dash import callback
from wordcloud import WordCloud
import plotly.express as px

ENTITIES_GRAPH_PATH = "entity_graph.pkl"
ENTITIES_IMPORTANCE_THRESHOLD = 0
ENTITIES_TITLE = "Entity Graph Visualization"
ENTITIES_ACTIVE_GRAPH = 'entity'

HASHTAGS_GRAPH_PATH = "hashtag_graph.pkl"
HASHTAGS_IMPORTANCE_THRESHOLD = 0
HASHTAGS_TITLE = "Hashtag Graph Visualization"
HASHTAGS_ACTIVE_GRAPH = 'hashtag'

def retrieve_params() -> tuple[str, str, str, str]:
    return ENTITIES_GRAPH_PATH, ENTITIES_IMPORTANCE_THRESHOLD, ENTITIES_TITLE, ENTITIES_ACTIVE_GRAPH
    #return HASHTAGS_GRAPH_PATH, HASHTAGS_IMPORTANCE_THRESHOLD, HASHTAGS_TITLE, HASHTAGS_ACTIVE_GRAPH

USERS_IMPORTANCE_THRESHOLD = 0
EDGES_LABEL_THRESHOLD = 12
NODE_LABEL_THRESHOLD = 10

USERS_COLOR = 'red'

def import_graph(path:str, user_threshold:float, entity_threshold:float) -> ig.Graph:
    with open(path, "rb") as f:
        graph = pickle.load(f)
    
    nodes_to_delete = set(graph.vs.select(lambda v: v["type"] == 0 and v["importance"] < user_threshold or
                                                    v["type"] == 1 and v["importance"] < entity_threshold)
                                        .indices)
        
    graph.delete_vertices(list(nodes_to_delete))
    return graph

def filter_graph(g, user_importance_range:tuple[float, float], 
                 entity_importance_range:tuple[float, float], 
                 sentiment_thresholds:tuple[float, float], 
                 sentiment_outside:bool):
    
    vertices_to_show = []
    edges_to_show = []

    user_importance_lower, user_importance_upper = user_importance_range
    entity_importance_lower, entity_importance_upper = entity_importance_range
    sentiment_lower_threshold, sentiment_upper_threshold = sentiment_thresholds

    for v in g.vs():
        if v["type"] == 0 and user_importance_lower <= v["importance"] <= user_importance_upper:
            vertices_to_show.append(v.index)
        elif v["type"] == 1 and entity_importance_lower <= v["importance"] <= entity_importance_upper:
            if sentiment_outside:
                if v["sentiment"] <= sentiment_lower_threshold or v["sentiment"] >= sentiment_upper_threshold:
                    vertices_to_show.append(v.index)
            else:
                if sentiment_lower_threshold <= v["sentiment"] <= sentiment_upper_threshold:
                    vertices_to_show.append(v.index)
    
    for e in g.es():
        source, target = e.tuple
        if source in vertices_to_show and target in vertices_to_show:
            edges_to_show.append(e.tuple)
    
    return vertices_to_show, edges_to_show

def compute_color_by_sentiment(sentiment:float) -> str:
    if sentiment is None:
        return 'grey'
    
    assert -1 <= sentiment <= 1, "Sentiment must be in the range [-1, 1]"

    rgb = [0, 0, 0]
    if sentiment < 0:
        rgb[0] = 1 + sentiment
        rgb[1] = 1 + sentiment
        rgb[2] = 1
    else:
        rgb[0] = 1 
        rgb[1] = 1 - sentiment
        rgb[2] = 1 - sentiment

    color = f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})'
    return color
    
def compute_width_by_importance(importance:float) -> float:
    return 0.25 + importance * 2

def create_figure(vertices:list[tuple[int, int]], edges:list[tuple[int, int]], g, users_color:str=USERS_COLOR):
    
    global active_graph
    
    subgraph = g.induced_subgraph(vertices)

    vertices = [v.index for v in subgraph.vs()]
    edges = [e.tuple for e in subgraph.es()]

    layout = subgraph.layout("bipartite")

    for i, coord in enumerate(layout.coords):
        subgraph.vs()[i]["x"] = coord[0]
        subgraph.vs()[i]["y"] = coord[1]
    
    Xn = [subgraph.vs()[v]["x"] for v in vertices]
    Yn = [subgraph.vs()[v]["y"] for v in vertices]
    node_sizes = [20 + subgraph.vs()[v]["importance"] * 70 for v in vertices]  # Dimensione nodi in base all'importanza
    
    Xe = []
    Ye = []

    for e in edges:
        Xe += [subgraph.vs[e[0]]["x"], subgraph.vs[e[1]]["x"], None]
        Ye += [subgraph.vs[e[0]]["y"], subgraph.vs[e[1]]["y"], None]

    # Calcola le coordinate dei marker spostate verso il nodo in alto
    X_edge_weights = [(0.6 * subgraph.vs[e[0]]["x"] + 0.4 * subgraph.vs[e[1]]["x"]) for e in edges]
    Y_edge_weights = [(0.6 * subgraph.vs[e[0]]["y"] + 0.4 * subgraph.vs[e[1]]["y"]) for e in edges]

    edge_trace = go.Scatter(x=Xe, y=Ye, 
                            mode='lines',
                            line=dict(width=0.3, 
                                      color='grey'), 
                            name='',
                            hoverinfo='none',
                            showlegend=False
                            )
    
    #edge_widths = [compute_width_by_importance(subgraph.es()[subgraph.get_eid(e[0], e[1])]["weight"]) for e in edges]
    #edge_trace['line']['width'] = edge_widths
    
    show_nodes_labels = sum([1 for v in subgraph.vs() if v['type']]) <= NODE_LABEL_THRESHOLD and sum([1 for v in subgraph.vs() if not v['type']]) <= NODE_LABEL_THRESHOLD
    show_edge_labels = subgraph.ecount() <= EDGES_LABEL_THRESHOLD and show_nodes_labels

    edge_weights_trace = go.Scatter(x=X_edge_weights, y=Y_edge_weights, 
                                    mode='markers+text' if show_edge_labels else "text",
                                    marker=dict(size=3, color='black'),
                                    textposition='top center',
                                    text=[subgraph.es()[subgraph.get_eid(e[0], e[1])]['weight'] for e in edges] if show_edge_labels else None,
                                    hovertemplate=[f"Mentions: {subgraph.es()[subgraph.get_eid(e[0], e[1])]['weight']}" for e in edges],
                                    name=''
                                    )

    node_colors = []
    for v in vertices:
        if subgraph.vs[v]["type"] == 0:
            node_colors.append(users_color)
        else:
            node_colors.append(compute_color_by_sentiment(subgraph.vs[v]["sentiment"]))

    if active_graph == ENTITIES_ACTIVE_GRAPH:
        hovertext = [f"Name: {subgraph.vs()[v]['name']}<br>Category: {subgraph.vs()[v]['category']}<br>Importance: {subgraph.vs()[v]['importance']:.4f}<br>Sentiment: {subgraph.vs()[v]['sentiment']:.4f}" if subgraph.vs()[v]['type'] == 1 
                    else f"Name: {subgraph.vs()[v]['name']}<br>Followers: {int(subgraph.vs()[v]['followers'])}<br>Importance: {subgraph.vs()[v]['importance']:.3f}"
                    for v in vertices]
    else:
        hovertext = [f"Name: {subgraph.vs()[v]['name']}<br>Importance: {subgraph.vs()[v]['importance']:.4f}<br>Sentiment: {subgraph.vs()[v]['sentiment']:.4f}" if subgraph.vs()[v]['type'] == 1 
                    else f"Name: {subgraph.vs()[v]['name']}<br>Followers: {int(subgraph.vs()[v]['followers'])}<br>Importance: {subgraph.vs()[v]['importance']:.3f}"
                    for v in vertices]
        
    node_trace = go.Scatter(
                x=Xn, 
        y=Yn, 
        mode='markers+text' if show_nodes_labels else "markers",
        marker=dict(
            size=node_sizes, 
            color=node_colors,
        ), 
        text=[subgraph.vs()[v]['name'] for v in vertices],
        hovertext=hovertext,
        hoverinfo='text',
        textposition='top center'
    )

    fig = go.Figure(data=[edge_trace, edge_weights_trace, node_trace], layout=go.Layout(
        showlegend=False, 
        hovermode='closest', 
        margin=dict(b=20, l=5, r=5, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    
    return fig

app = dash.Dash(__name__)

graph_path, importance_threshold, title, active_graph = retrieve_params()
graph = import_graph(graph_path, USERS_IMPORTANCE_THRESHOLD, importance_threshold)

from collections import Counter
import pickle
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
    
    html.Div([
        html.H1(title, style={'float':'left', 'textAlign': 'center', 'color': '#333', 'font-family': 'Arial'}),
        html.Div([
            html.Label('Graph:', style={'font-family': 'Arial', 'fontSize': '14px', 'margin-right': '10px'}),
            dcc.RadioItems(
                id='graph-selector',
                options=[
                    {'label': 'Entities Graph', 'value': ENTITIES_ACTIVE_GRAPH},
                    {'label': 'Hashtags Graph', 'value': HASHTAGS_ACTIVE_GRAPH}
                ],
                value=ENTITIES_ACTIVE_GRAPH,
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
        ], style={'float': 'right', 'width': '30%', 'margin-right': '10px', 'display': 'flex', 'align-items': 'center'})
    ], style={'overflow': 'hidden', 'padding-bottom': '1px', 'width': '100%', 'display': 'flex',
        'justifyContent': 'space-around', 'align-items': 'center'}),

    
    html.Div([
        html.Div([
            html.Div('Users visualized:', className='label'),
            html.Div(id='users-count', className='value')
        ], className='stat-item', style={'flex': '1', 'padding': '10px'}),
        html.Div([
            html.Div('Entities visualized:', className='label'),
            html.Div(id='entities-count', className='value')
        ], className='stat-item', style={'flex': '1', 'padding': '10px'}),
        html.Div([
            html.Div('Correlation:', className='label'),
            html.Div(id='correlation-value', className='value')
        ], className='stat-item', style={'flex': '1', 'padding': '10px'})
    ], id='user-entity-counts', style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'align-items': 'stretch',
        'backgroundColor': '#f9f9f9',
        'padding': '5px',
        'margin': '5px auto',
        'border': '1px solid black',
        'borderRadius': '5px',
        'width': '88%',
        'textAlign': 'center',
        'fontSize': '14px',
        'lineHeight': '1.2'
    }),

    dcc.Graph(id='graph', style={'height': '60vh', 'width': '89%', 'margin': 'auto', 'border': '1px solid black'}),
    
    html.Div([
        html.Div([
            html.Label('User Importance', style={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '14px'}),
            dcc.RangeSlider(
                id='user-threshold', min=USERS_IMPORTANCE_THRESHOLD, max=1, step=0.001, value=[USERS_IMPORTANCE_THRESHOLD, 1], 
                marks={i: str(i) for i in [0, 0.5, 1]},
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False,
            ),
        ], style={'width': '100%', 'padding': '0px 10px', 'box-sizing': 'border-box'}),
        html.Div([
            html.Label('Entity Importance', style={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '14px'}),
            dcc.RangeSlider(
                id='entity-threshold', min=importance_threshold, max=1, step=0.001, value=[importance_threshold, 1], 
                marks={i: str(i) for i in [0, 0.5, 1]},
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False,
            ),
        ], style={'width': '100%', 'padding': '0px 10px', 'box-sizing': 'border-box'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'align-items': 'stretch', 'textAlign': 'center', 'margin': '10px auto', 'width': '95%'}),

    html.Div([
        html.Label('Entity Sentiment', style={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '14px'}),
        dcc.RangeSlider(
            id='sentiment-threshold', min=-1, max=1, step=0.001, value=[-1, 1], 
            marks={i: str(i) for i in [-1, 0, 1]},
            tooltip={"placement": "bottom", "always_visible": True},
            allowCross=False,
        ),
        html.Div(id='sentiment-labels', style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '0 10px'})
    ], style={'width': '95%', 'padding': '10px', 'margin': '10px auto', 'textAlign': 'center', 'box-sizing': 'border-box'}),

    html.Div([
        dcc.Checklist(
            id='sentiment-outside',
            options=[
                {'label': 'Use sentiment values outside of interval', 'value': 'outside'}
            ],
            value=[]
        )
    ], style={'textAlign': 'center', 'font-family': 'Arial', 'fontSize': '14px', 'padding': '5px', 'width': '100%', 'margin': '5px auto', 'box-sizing': 'border-box'})  # Ridotto padding e marginatura
], style={'width': '80%', 'margin': '0 auto'}
)

@app.callback(
    [
        Output('graph', 'figure'),
        Output('user-entity-counts', 'children')
    ],
    [
        Input('graph-selector', 'value'),
        Input('user-threshold', 'value'),
        Input('entity-threshold', 'value'),
        Input('sentiment-threshold', 'value'),
        Input('sentiment-outside', 'value')
    ]
)
def update_graph_and_counts(selected_graph:str, 
                            user_importance_range:tuple[float, float],
                            entity_importance_range:tuple[float, float],
                            sentiment_thresholds:tuple[float, float],
                            sentiment_outside:str) -> tuple[go.Figure, str]:

    global graph
    global active_graph
    global graph_path
    global importance_threshold
    global title

    if selected_graph != active_graph:
        if selected_graph == ENTITIES_ACTIVE_GRAPH:
            graph_path, importance_threshold, title = ENTITIES_GRAPH_PATH, ENTITIES_IMPORTANCE_THRESHOLD, ENTITIES_TITLE
        else:
            graph_path, importance_threshold, title = HASHTAGS_GRAPH_PATH, HASHTAGS_IMPORTANCE_THRESHOLD, HASHTAGS_TITLE

        active_graph = selected_graph
        graph = import_graph(graph_path, USERS_IMPORTANCE_THRESHOLD, importance_threshold)

    sentiment_outside_val = 'outside' in sentiment_outside
    vertices, edges = filter_graph(graph, 
                                   user_importance_range, 
                                   entity_importance_range, 
                                   sentiment_thresholds, 
                                   sentiment_outside_val)

    user_count = sum(1 for v in graph.vs() if v["type"] == 0 and 
                                          user_importance_range[0] <= v["importance"] <= user_importance_range[1])
    
    if sentiment_outside_val:
        entities = [v for v in graph.vs() if v["type"] == 1 and 
                                            entity_importance_range[0] <= v["importance"] <= entity_importance_range[1] and
                                            (v["sentiment"] <= sentiment_thresholds[0] or v["sentiment"] >= sentiment_thresholds[1])]
    else:
        entities = [v for v in graph.vs() if v["type"] == 1 and 
                                            entity_importance_range[0] <= v["importance"] <= entity_importance_range[1] and
                                            (sentiment_thresholds[0] <= v["sentiment"] <= sentiment_thresholds[1])]
        
    average_sentiment = np.mean([entity["sentiment"] for entity in entities]) if len(entities) > 0 else 0
    average_polarity = np.mean([abs(entity["sentiment"]) for entity in entities]) if len(entities) > 0 else 0
    
    counts = f"Users: {user_count}, Entities: {len(entities)}, Average sentiment: {average_sentiment:.3f}, Average polarity: {average_polarity:.3f}"
    
    return create_figure(vertices, edges, graph), counts



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
