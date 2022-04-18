import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations



####################### Colors 
COLORS = ['#e06666', '#e88b81', '#f0ad9a', '#f8ceb2', '#ffeeca', '#c6cfbf', '#8fb0ae', '#58919d', '#20718b']

CAT_COLORS = ['#20718b', '#7DB082', '#FAEC73', '#ECA052', '#e06666']
CONTRAST_COLORS = [COLORS[i] for i in range(len(COLORS)) if (i%2) == 0]

DIV_COLORS = ['#e06666', '#e68d8d', '#ebb1b1', '#efd3d3', '#f4f4f4', '#bed3da', '#8ab2bf', '#5592a5', '#20718b']

UMP_COLORS = ['#22446D', '#FC9E4F', '#AB2346' ,'#6ABB5D']

#DEFAULT_PALETTE = sns.color_palette(COLORS)
#CONTRAST_PALETTE = sns.color_palette(CONTRAST_COLORS)
#DIVERGENT_PALETTE = sns.color_palette(DIV_COLORS)
#CAT_PALETTE = sns.color_palette(CAT_COLORS)
#UMP_PALETTE = sns.color_palette(UMP_COLORS)


#DIV_CMAP = LinearSegmentedColormap.from_list("div_colors", DIV_COLORS)
#CAT_CMAP = LinearSegmentedColormap.from_list("cat_colors", CAT_COLORS)
#UMP_CMAP = LinearSegmentedColormap.from_list("ump_colors", UMP_COLORS)


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


def basket_recommendations(i, sim_df, cols, customer, k=5):
    
    ix = sim_df.loc[:,i].to_numpy().argpartition(range(-1,-k,-1))
    closest = sim_df.columns[ix[-1:-(k+2):-1]]
    closest = closest.drop(i, errors='ignore')
    
    recs = pd.DataFrame(closest).merge(cols)
    recs = recs.loc[recs.CustomerID != customer]
    
    return recs.head(k)
def create_basket(items, customer=99999999):
    """
    create an "invoice"
    """
    return pd.DataFrame(data={'InvoiceNo': [0], 'CustomerID':[customer], 'Description': [(',').join(items)], 'combined': [(' ').join(items)],  })

new_basket = create_basket(['towel design', 'alarm clock bakelike']) 
#pd.concat([df_inv, new_basket])


def vectorize_invoices(df):
    tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,3)
                                    for c in combinations(s.split(','), r=i)))

    tfidf_matrix = tf.fit_transform(df['Description'])
    
    return tfidf_matrix

## 
## tfidf_matrix = vectorize_invoices(merged_inv, 'Description')
## 

def make_cosine_matrix(mx, idx ):
    cosine_sim = cosine_similarity(mx)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=idx, columns=idx)
    return cosine_sim_df

## 
## cosine_sim_df = make_cosine_matrix(tfidf_matrix, merged_inv['InvoiceNo'])
## 


def get_similar_baskets(items, invoices, customerid=-99999999, k=5, idx='InvoiceNo'):
    new_basket = create_basket(items, customerid)
    
    inv_ = pd.concat([invoices, new_basket])
    

    
    ## vectorize invoices
    tfidf_matrix = vectorize_invoices(inv_)
    
    ## make similarity matrix
    similarity_matrix = make_cosine_matrix(tfidf_matrix, inv_[idx])
    
    i = new_basket['InvoiceNo'][0]
    
    ix = similarity_matrix.loc[:,i].to_numpy().argpartition(range(-1,-k,-1))

    closest = similarity_matrix.columns[ix[-1:-(k+2):-1]]
    closest = closest.drop(i, errors='ignore')
    
    recs = pd.DataFrame(closest).merge(inv_[['InvoiceNo', 'Description', 'CustomerID']])
    recs = recs.loc[recs.CustomerID != customerid]
    
    
    return recs

def get_product_recs(similar_baskets, items):
    basket_contents = list(set(','.join(similar_baskets['Description'].tolist()).split(',')))
    diff = list(set(basket_contents) - set(items))
    
    return diff





################ Data Wrangling ################

## Load Data

## Retail
nxload_rt = read_json_file(DATA_PATH + 'nxdump_rt.json')
nodes_rt = [node for node in nxload_rt.nodes()]

pos_rt = nx.spring_layout(nxload_rt, k=5, weight='weight')


edge_trace_rt, node_trace_rt = make_graph(nxload_rt, pos_rt, nodes_rt)
fig_rt = make_figure(edge_trace_rt, node_trace_rt, "Graph of Most Frequent Retail Customer Purchases")

## Wholesale
nxload_ws = read_json_file(DATA_PATH + 'nxdump_ws.json')
nodes_ws = [node for node in nxload_ws.nodes()]

pos_ws = nx.spring_layout(nxload_ws, k=5, weight='weight')


edge_trace_ws, node_trace_ws = make_graph(nxload_ws, pos_ws, nodes_ws)
fig_ws = make_figure(edge_trace_ws, node_trace_ws, "Graph of Most Frequent Wholesale Customer Purchases")

################ products ################

LIMIT_INVOICES = 200

df = pd.read_csv(DATA_PATH+'data_cleaned.csv')
df = df.loc[df['IsCancelled'] == False]

invoice_list = df['InvoiceNo'].unique()[0:LIMIT_INVOICES]

df_inv = df.loc[df['InvoiceNo'].isin(invoice_list)]


products_list = df_inv['Description'].unique()
new_items = ['towel design']


products_dropdown = dcc.Dropdown(
       id='products_dropdown',
       options=products_list,
       multi=True,
       value=new_items
   )


################ TFIDF ################



product_recs_div = html.Div([])

################ APP ################

app = dash.Dash(__name__)

server = app.server


app.layout = html.Div([
    html.Div([

        html.Div([
            html.Div([
                html.Div([
                    html.H2("Select Products"),
                    products_dropdown
                ], className='col'),
                html.Div([
                    html.H2("You might like:"),
                    html.H3("Other Customers Also Bought:"),
                    html.Div(id="product_recs_div"),
                ], className='col'),

            ], className='row'),
            html.Div([], className='row'),
        ], className="card"),


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

################ Callbacks ################


@app.callback(
    Output('product_recs_div', 'children'),
    Input('products_dropdown', 'value'),

)
def show_recommendations(new_items):

    similar_baskets = get_similar_baskets(new_items, df_inv )
    product_recs = get_product_recs(similar_baskets, new_items)

    product_list_result = html.Ul(
        [html.Li(i) for i in product_recs[0:10]]
    )

    return html.Div(product_list_result)


if __name__ == '__main__':
    app.run_server(debug=True)
