import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import scipy.stats as stats
from math import ceil
from datetime import timedelta, datetime
import os
import re
import itertools
import collections

import warnings
warnings.filterwarnings('ignore')



# pip install networkx
import networkx as nx
from networkx.readwrite import json_graph
import json 

import plotly.offline as py
import plotly.graph_objects as go

# pip install pydot
# pip install graphviz

import pydot
import graphviz



import dash
from dash import dash_table, html, dcc

from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import plotly.graph_objs as go


####################### Colors 
COLORS = ['#e06666', '#e88b81', '#f0ad9a', '#f8ceb2', '#ffeeca', '#c6cfbf', '#8fb0ae', '#58919d', '#20718b']

CAT_COLORS = ['#20718b', '#7DB082', '#FAEC73', '#ECA052', '#e06666']
CONTRAST_COLORS = [COLORS[i] for i in range(len(COLORS)) if (i%2) == 0]

DIV_COLORS = ['#e06666', '#e68d8d', '#ebb1b1', '#efd3d3', '#f4f4f4', '#bed3da', '#8ab2bf', '#5592a5', '#20718b']

UMP_COLORS = ['#22446D', '#FC9E4F', '#AB2346' ,'#6ABB5D']

DEFAULT_PALETTE = sns.color_palette(COLORS)
CONTRAST_PALETTE = sns.color_palette(CONTRAST_COLORS)
DIVERGENT_PALETTE = sns.color_palette(DIV_COLORS)
CAT_PALETTE = sns.color_palette(CAT_COLORS)
UMP_PALETTE = sns.color_palette(UMP_COLORS)


DIV_CMAP = LinearSegmentedColormap.from_list("div_colors", DIV_COLORS)
CAT_CMAP = LinearSegmentedColormap.from_list("cat_colors", CAT_COLORS)
UMP_CMAP = LinearSegmentedColormap.from_list("ump_colors", UMP_COLORS)


SHOW_PLOTS = True
SAVE_PLOTS = True
RANDOM_STATE = 9

IMG_PATH = 'imgs/'
DATA_PATH = 'data/'


optionslist = ["Foo", "Bar", "Baz"]







################ Functions 

def save_fig(title, fig):
    if SAVE_PLOTS == True:
        fn = IMG_PATH + title.replace(' ','-') + '.png'
        fig.savefig(fn, bbox_inches='tight', transparent=True)
    return 

def make_graph(G, pos, nodes):
    """
    Create graph data structure
    """
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text = "",
        textfont=dict(
            family="sans serif",
            size=10
        ),
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=10,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{nodes[node]}: {str(len(adjacencies[1]))} connections')

    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = node_text
    
    return edge_trace, node_trace
    
def make_figure(edge_trace, node_trace, title):
    """
    Create graph plot using edges and nodes given
    """
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )


    #fig.show()
    return fig
    

def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)


################ Aesthetics 


sns.set(style="white")
sns.set_context("paper")
sns.set_palette(DEFAULT_PALETTE)
plt.rcParams['figure.dpi'] = 128







################ Data Wrangling ################

## Load Data

## Retail
nxload_rt = read_json_file(DATA_PATH + 'nxdump_rt.json')
nodes_rt = [node for node in nxload_rt.nodes()]
pos_rt = nx.nx_pydot.graphviz_layout(nxload_rt, prog="neato")


edge_trace_rt, node_trace_rt = make_graph(nxload_rt, pos_rt, nodes_rt)
fig_rt = make_figure(edge_trace_rt, node_trace_rt, "Graph of Retail Customer Purchases")

## Wholesale
nxload_ws = read_json_file(DATA_PATH + 'nxdump_ws.json')
nodes_ws = [node for node in nxload_ws.nodes()]
pos_ws = nx.nx_pydot.graphviz_layout(nxload_ws, prog="neato")


edge_trace_ws, node_trace_ws = make_graph(nxload_ws, pos_ws, nodes_ws)
fig_ws = make_figure(edge_trace_ws, node_trace_ws, "Graph of Wholesale Customer Purchases")


dropdown_cc = dcc.Dropdown(
       id='cc_drop',
       options=optionslist,
       multi=False,
       value='Foo'
   )

c = pd.read_csv('data/test.csv')

################ APP ################

app = dash.Dash(__name__)

server = app.server


app.layout = html.Div([

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig_ws)
                ], className="col"),
                html.Div([
                    dcc.Graph(figure=fig_rt)
                ], className="col"),
            ], className="row"),
        ], className='card'),
    ], className="container"),




])




if __name__ == '__main__':
    app.run_server(debug=True)
