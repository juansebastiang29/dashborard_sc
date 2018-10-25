# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Output
from dash.dependencies import Input
from DataWrangling import *
import appConfig as appconfig

# Setup the app
app = dash.Dash('MerqueoModelDashboard')


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
    'text': '#FEF3EE',
    'text_header': '#a7bfd0',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

backgroundGraphs = {'background': '#171b1e'}

df_pasillo_cat = df.groupby(["Pasillo","Categoria"],as_index=False).sum()[['Pasillo','Categoria','cuenta_gr']]

app.layout = html.Div(
    html.Div(
    [html.Div(
          [
              html.H1("Model Dashboard",
                      className='eight columns',style=dict(color='#ffff', size='18')),
              html.Div(id='none', children=[],
                       style={'display': 'none'})
          ], className="row header"
      ),
    html.Div([
        html.Div([
            html.H5('', id='ventas_text', className='twelve columns')
            ],className="row"
        ),
        html.Div([html.H5('', id='ventas_ciudades', className='eight columns',
                    style={'text-align': 'left'}),
            html.H5('', id='year_text', className='four columns', style={'text-align': 'right'})
                  ],className="row"
        ),
        html.Div([html.P('Filter by City:'),
            dcc.RadioItems(id="cityselector",
                           options=ciudad_options,
                           value='All',
                           labelStyle={'display': 'inline-block'})]),
        html.Div([
            html.P('Filter by date:', style={'margin-top': '5'}),
            html.Div([dcc.RangeSlider(id='year_slider',
                            min= 0,max=list(marks_slider.keys())[-1],
                            value= [0, list(marks_slider.keys())[-1]],
                            marks= marks_slider,)],
                     style={'margin-top': '20',
                            'margin-right':'40',
                            'margin-left':'40'}),
        ], style={'margin-top': '20'}),
        html.Div([
            html.Div([
                    html.P('Filter by Section:'),
                    dcc.RadioItems(id="pasillo_selector",
                           options=[{'label':'All', 'value':'All'},
                                    {'label':'Custom', 'value':'Custom'}],
                           value='All',
                           labelStyle={'display': 'inline-block'}),
                    dcc.Dropdown(id='pasillo_options',
                                 options=pasillo_options,
                                 multi=True,
                                 value=[])
            ],className="six columns")
        ], className="row", style={'margin-top': '30'}),
        html.Div(children='Dash: A web application framework for Python.',
                 className='col s12 m6 l6',
                 style={'color':'#000000',
                        'margin-top': '25'}),
        html.Div([
            indicator(
                    "#00cc96",
                    "Total Orders",
                    "TotalDeliveries",
                ),
            indicator(
                    "#119DFF",
                    "Delivered",
                    "prct_delivered",
                ),
            indicator(
                    "#EF553B",
                    "Total Sales Delivered orders",
                    "total_sales_delv",
                )],className="row"),
        html.Div([
            html.Div(
                [
                    dcc.Graph(id='cantidad_vendida_serie',
                              animate=False)
                ],className='eight columns',
                style={'margin-top': '20'}),
            html.Div(
                [
                    dcc.Graph(id='individual_graph_1')
                ],className='four columns',
                style={'margin-top': '20'})
            ],className="row"),
        html.Div([
            html.Div(
                [
                    dcc.Graph(id='sales_series',
                              animate=False)
                ],className='eight columns',
                style={'margin-top': '20'}),
            html.Div(
                [
                    dcc.Graph(id='individual_graph_2')
                ],className='four columns',
                style={'margin-top': '20'})
            ],className='row')
        ],className="row", style={'margin-right':'20', 'margin-left':'20'}),
        html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet"),
        html.Link(href="https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css",rel="stylesheet")
    ],
    style={'font' : dict(family = "Helvetica",
                       size = 14,
                       color = '#CCCCCC',)}
),style={'backgroundColor':'#ffff'})

# Create callbacks


# Radio -> multi
@app.callback(Output('pasillo_options', 'value'),
              [Input('pasillo_selector', 'value')])
def display_status(selector):

    if selector == 'Custom':
        return ['Frutas']
    else:
        return []

# # Radio -> multi
# @app.callback(Output('categoria_options', 'value'),
#               [Input('categoria_selector', 'value'),
#                Input("pasillo_options","value"),
#                Input("pasillo_selector","value")])
# def display_status(pasillo_options, pasillo_selector, selector):
#     if pasillo_selector == 'Custom':
#         list = df_pasillo_cat[df_pasillo_cat.Pasillo.isin(pasillo_options)].Categoria.unique().tolist()
#     else:
#         list = df_pasillo_cat[df_pasillo_cat.Pasillo.str.contains(pasillo_options)].Categoria.unique().tolist()
#     if selector == 'Custom':
#         return list
#     else:
#         return []

# Selectors -> well text
@app.callback(Output('ventas_text', 'children'),
              [Input('year_slider', 'value')])
def update_well_text(year_slider):
    return "Total Sales %dB" % (df[marks_slider[year_slider[0]]:marks_slider[year_slider[1]]]\
                                       ['TotalPagado'].sum()/appconfig.sales_scale)

@app.callback(Output('year_text', 'children'),
              [Input('year_slider', 'value')])
def update_well_text(year_slider):
    return " | ".join([marks_slider[e] for e in year_slider])

@app.callback(Output('ventas_ciudades', 'children'),
              [Input('year_slider', 'value')])
def update_well_text(year_slider):
    difagg = df[marks_slider[year_slider[0]]:marks_slider[year_slider[1]]]\
                .groupby("Ciudad").sum()[["TotalPagado"]]
    return " | ".join(["%s %dB" % (e[0], e[1] / appconfig.sales_scale)\
                                             for e in zip(difagg.index, difagg.values)])

@app.callback(Output('TotalDeliveries', 'children'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def total_deliveries_indicator_callback(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options).groupby("estadoOrden")

    return "{}K".format(str(np.round(dff.sum()["cuenta_gr"].sum()/appconfig.orders_scale, 1)))

@app.callback(Output('prct_delivered', 'children'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def prct_deliveries_indicator_callback(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options).groupby("estadoOrden")
    difagg = dff.sum()[["cuenta_gr"]] / \
             dff.sum()[["cuenta_gr"]].sum()
    return "{}%".format(str(np.round(difagg.loc['Delivered'].values[0], 3) * 100))

@app.callback(Output('total_sales_delv', 'children'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def prct_deliveries_indicator_callback(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options).groupby("estadoOrden")

    return "%dB" % (dff.sum()["TotalPagado"].sum()/appconfig.sales_scale)


@app.callback(Output('cantidad_vendida_serie', 'figure'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def update_y_timeseries(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options)
    df_sb_b = dff.resample('D').sum()

    return {'data': [go.Scatter(x=df_sb_b.index,
                                y=df_sb_b.cantidadVendida.values,
                                line=dict(color = '#236bb2',
                                          width = 2,))],
            'layout': go.Layout(
                title='Daily Sold Quantities',
                plot_bgcolor = backgroundGraphs['background'],
                paper_bgcolor = backgroundGraphs['background'],
                xaxis=dict(title='Time in Days',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                yaxis=dict(title='Sold Quantities',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                font = dict(color='#e8e9e9')
            )}

@app.callback(Output('individual_graph_1', 'figure'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def update_y_timeseries(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options)

    bin_val = np.histogram(np.log10(dff[dff.cantidadVendida>0].cantidadVendida.values))

    print("histogram",len(dff.cantidadVendida.values))
    return {'data': [go.Bar(x=bin_val[1],
                            y=bin_val[0],
                            marker=go.Marker(
                                color='#66b3ff'))],
            'layout': go.Layout(
                title='Histogram Sold Quantities (Log base 10)',
                plot_bgcolor=backgroundGraphs['background'],
                paper_bgcolor=backgroundGraphs['background'],
                xaxis=dict(title='Log 10 Sold Quantity',
                           zerolinecolor='#909395'),
                yaxis=dict(title='Frequency',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                font=dict(color='#e8e9e9')

            )}

@app.callback(Output('sales_series', 'figure'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def update_sales_timeseries(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options)
    df_sb_b = dff.resample('D').sum()

    return {'data': [go.Scatter(x=df_sb_b.index,
                                y=np.log10(df_sb_b.TotalPagado.values),
                                line=dict(color = '#236bb2',
                                          width = 2,))],
            'layout': go.Layout(
                title='Daily Sales',
                plot_bgcolor = backgroundGraphs['background'],
                paper_bgcolor = backgroundGraphs['background'],
                xaxis=dict(title='Time in Days',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                yaxis=dict(title='Log 10 Sales',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                font = dict(color='#e8e9e9')
            )}

@app.callback(Output('individual_graph_2', 'figure'),
              [Input('year_slider', 'value'),
               # Input('categoria_selector','value'),
               Input('pasillo_options','value'),
               Input('pasillo_selector','value'),
               Input('cityselector', 'value')])
def update_y_timeseries(year_slider, pasillo_options, pasillo_selector, cityselector):
    dff = subset_dataframe(df=df,
                           location=cityselector,
                           filter_date=year_slider,
                           sections_selector=pasillo_selector,
                           section_options=pasillo_options)

    bin_val = np.histogram(np.log10(dff[dff.TotalPagado>0].TotalPagado.values),bins=50)

    print("histogram",len(dff.cantidadVendida.values))
    return {'data': [go.Bar(x=bin_val[1],
                            y=bin_val[0],
                            marker=go.Marker(
                                color='#66b3ff'))],
            'layout': go.Layout(
                title='Histogram Total Sales (Log base 10)',
                plot_bgcolor=backgroundGraphs['background'],
                paper_bgcolor=backgroundGraphs['background'],
                xaxis=dict(title='Log 10 Sales',
                           zerolinecolor='#909395'),
                yaxis=dict(title='Frequency',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                font=dict(color='#e8e9e9')

            )}

if __name__ == '__main__':
    app.run_server(debug=True,
                   host="0.0.0.0",
                   threaded=True)
