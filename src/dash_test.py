# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Setup the app
app = dash.Dash('MerqueoModelDashboard')

# Load the css
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})


external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js',
               'https://pythonprogramming.net/static/socialsentiment/googleanalytics.js']
for js in external_js:
    app.scripts.append_script({'external_url': js})

sentiment_colors = {-1:"#EE6055",
                    -0.5:"#FDE74C",
                     0:"#FFE6AC",
                     0.5:"#D0F2DF",
                     1:"#9CEC5B",}


app_colors = {
    'background': '#0C0F0A',
    'text': '#FFFFFF',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

df = pd.read_csv("../Data/AGG_MERQUEO.csv", encoding='latin1')
df['FechaCreaciónOrden'] = pd.to_datetime(df.FechaCreaciónOrden)
df.set_index("FechaCreaciónOrden", inplace=True)

def generate_ts(df,location="Bogotá"):
    df_sb = df[df.Ciudad.str.contains(location)][['cantidadVendida']]
    return df_sb.resample('D').sum()


app.layout = html.Div(
    [ html.Div(className='container-fluid',
             children=[html.H2('Tablero Modelo Predicción', style={'color':"#CECECE"}),
                       dcc.Input(id='sentiment_term',
                                 value='Escribe algo...',
                                 type='text',
                                 style={'color':app_colors['text']}),],
             style={'width':'98%','margin-left':10, 'margin-right':10, 'max-width':50000}),
      html.Div(children='Dash: A web application framework for Python.',
                className='col s12 m6 l6', style={'color':app_colors['text']}),
      html.Div(className='row', children=[html.Div(dcc.Graph(id='bogota-serie', animate=False), className='col s12 m6 l6'),
                                          html.Div(dcc.Graph(id='medellin-serie', animate=False), className='col s12 m6 l6')])

    # html.Div(className='row',children=[
    #         dcc.Graph(id='x-time-series'),dcc.Graph(id='y-time-series')])
    ], style={'backgroundColor': app_colors['background'],
              'margin-top':'-30px',
              'height':'2000px',})


@app.callback(
    dash.dependencies.Output('bogota-serie', 'figure'),
    [dash.dependencies.Input('sentiment_term', 'value')])
def update_y_timeseries(value):
    df_sb_b = generate_ts(df, location="Bogotá")
    return {'data': [go.Scatter(x=df_sb_b.index,
                                y=df_sb_b.cantidadVendida.values,
                                line=dict(color = (app_colors['sentiment-plot']),width = 1,))],
            'layout': go.Layout(
                title='Cantidad vendida Bogotá',
                font={'color':app_colors['text']},
                plot_bgcolor = app_colors['background'],
                paper_bgcolor = app_colors['background'])}


@app.callback(
    dash.dependencies.Output('medellin-serie', 'figure'),
    [dash.dependencies.Input('sentiment_term', 'value')])
def update_y_timeseries(value):
    df_sb_b = generate_ts(df, location="Medellín")
    return {'data': [go.Scatter(x=df_sb_b.index,
                                y=df_sb_b.cantidadVendida.values,
                                line=dict(color = (app_colors['sentiment-plot']),width = 1,))],
            'layout': go.Layout(
                title='Cantidad vendida Medellin',
                font={'color':app_colors['text']},
                plot_bgcolor = app_colors['background'],
                paper_bgcolor = app_colors['background'])}

if __name__ == '__main__':
    app.run_server(debug=True,
                   host="0.0.0.0")
