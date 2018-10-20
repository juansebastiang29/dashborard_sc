# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Output
from dash.dependencies import Input

# Setup the app
app = dash.Dash('MerqueoModelDashboard')

app.css.append_css({'external_url': "https://cdn.rawgit.com/plotly/dash-app-stylesheets/"
                                    "2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css"})  # noqa: E501
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
    'text': '#FFFFFF',
    'text_header': '#a7bfd0',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

backgroundGraphs = {'background': '#0C0F0A'}

df = pd.read_csv("../Data/AGG_MERQUEO.csv", encoding='latin1')
df['FechaCreaciónOrden'] = pd.to_datetime(df.FechaCreaciónOrden)

df.set_index("FechaCreaciónOrden", inplace=True,drop=False)

ciudad_options = [{'label': str(city),
                        'value': str(city)}
                       for city in ["All"]+sorted(df.Ciudad.unique().tolist()+["All"])]

categoria_options = [{'label': str(categoria), 'value': str(categoria)}\
                      for categoria in ["All"]+sorted(df.Categoria.unique().tolist())]

pasillo_options = [{'label': str(pasillo), 'value': str(pasillo)}\
                      for pasillo in ["All"]+sorted(df.Pasillo.unique().tolist())]

marks_slider = dict(enumerate(df['month_year'].unique()))
df_pasillo_cat = df.groupby(["Pasillo","Categoria"],as_index=False).sum()[['Pasillo','Categoria','cueta_gr']]

def subset_dataframe(df,location,filter_date, category, section):
    df_sb = df[marks_slider[filter_date[0]]:marks_slider[filter_date[1]]]
    df_sb = df[df.Ciudad.str.contains(location) &
               df.Pasillo.str.contains(section) &
               df.Categoria.str.contains(category)][['cantidadVendida']]
    return df_sb

def generate_ts(df,location="Bogotá"):
    df_sb = df[df.Ciudad.str.contains(location)][['cantidadVendida']]
    return df_sb.resample('D').sum()

# returns top indicator div
def indicator(color, text, id_value):
    return html.Div(
        [

            html.P(
                text,
                className="twelve columns indicator_text"
            ),
            html.P(
                id=id_value,
                className="indicator_value"
            ),
        ],
        className="four columns indicator",

    )

app.layout = html.Div(
    [html.Div(
          [
              html.H1("Tablero de Control",
                      className='eight columns',style=dict(color='#CCCCCC', size='14')),
              html.Div(id='none', children=[],
                       style={'display': 'none'})
          ], className="row header"
      ),
    html.Div([
        html.H5('',id='ventas_text',className='two columns'),
        html.H5('',id='ventas_ciudades',className='eight columns',
                style={'text-align': 'center'}),
        html.H5('', id='year_text', className='two columns',style={'text-align':'right'})
    ], className='row'),
    html.Div([html.P('Filter by City:'),
        dcc.RadioItems(id="cityselector",
                       options=ciudad_options,
                       value=[],
                       labelStyle={'display': 'inline-block'})]),
    html.Div([
        html.P('Filter by date:', style={'margin-top': '5'}),
        html.Div([dcc.RangeSlider(id='year_slider',
                        min=0, max=list(marks_slider.keys())[-1],
                        value=[0, list(marks_slider.keys())[-1]],
                        marks=marks_slider)],
                 style={'margin-top': '20', 'margin-right':'40', 'margin-left':'40'}),
    ], style={'margin-top': '20'}),
    html.Div([
        html.Div([
                html.P('Filter by Section:'),
                dcc.RadioItems(id="pasillo_selector",
                       options=[{'label':'All','value':''},
                                {'label':'Custom','value':'Custom'}],
                       value='All',
                       labelStyle={'display': 'inline-block'}),
                dcc.Dropdown(id='pasillo_options',
                             options=pasillo_options, multi=True, value=[])
        ],className="six columns"),
        html.Div([
                html.P('Filter by Category:'),
                dcc.RadioItems(id="categoria_selector",
                       options=[{'label':'All','value':''},
                                {'label':'Custom','value':'Custom'}],
                       value='All',
                       labelStyle={'display': 'inline-block'}),
                dcc.Dropdown(id='categoria_options',
                             options=categoria_options,
                             multi=True,
                             value=[])
        ],className='six columns')
    ], className="row", style={'margin-top': '30'}),
    html.Div(children='Dash: A web application framework for Python.',
             className='col s12 m6 l6',
             style={'color':app_colors['text'],
                    'margin-top': '25'}),
    html.Div([
            indicator(
                "#00cc96",
                "Total Deliveries",
                "TotalDeliveries",
            ),
            indicator(
                "#119DFF",
                "Delivered",
                "middle_opportunities_indicator",
            ),
            indicator(
                "#EF553B",
                "Sale Deliveries",
                "right_opportunities_indicator",
            )],className="row"),
    html.Div(className='row',children=[html.Div(dcc.Graph(id='bogota-serie',
                                                          animate=False),className='col s12 m6 l6'),
                                       html.Div(dcc.Graph(id='medellin-serie',
                                                          animate=False),className='col s12 m6 l6')],)
    ],
    style={'font' : dict(family = "Helvetica",
                       size = 12,
                       color = '#CCCCCC',),
           'margin-right':'20', 'margin-left':'20'}
)

# Create callbacks

@app.callback(Output('TotalDeliveries', 'children'),
              [Input('year_slider', 'value'),
               Input('categoria_selector','value'),
               Input('pasillo_selector','value'),
               Input('cityselector','value')])
def total_deliveries_indicator_callback(year_slider):
    dff = subset_dataframe(df=df,
                           filter_date=year_slider,
                           category=categoria_selector,
                           section=pasillo_options).groupby("estadoOrden")
    difagg = dff.sum()[["count_gr"]] / \
             dff.sum()[["count_gr"]].sum()
    lista_estados_ordenes = list(zip(difagg.index, difagg.values))
    indicador = lista_estados_ordenes[2]
    return "{}%".format(str(np.round(indicador[1], 3)[0] * 100))

# Selectors -> well text
@app.callback(Output('ventas_text', 'children'),
              [Input('year_slider', 'value')])
def update_well_text(year_slider):
    return "%dMM" % (df[marks_slider[year_slider[0]]:marks_slider[year_slider[1]]]\
                                       ['TotalPagado'].sum()/1000000000)

@app.callback(Output('year_text', 'children'),
              [Input('year_slider', 'value')])
def update_well_text(year_slider):
    return " | ".join([marks_slider[e] for e in year_slider])

@app.callback(Output('ventas_ciudades', 'children'),
              [Input('year_slider', 'value')])
def update_well_text(year_slider):
    difagg = df[marks_slider[year_slider[0]]:marks_slider[year_slider[1]]]\
                .groupby("Ciudad").sum()[["TotalPagado"]]
    return " | ".join(["%s %dMM" % (e[0], e[1] / 1000000000)\
                                             for e in zip(difagg.index, difagg.values)])


# Radio -> multi
@app.callback(Output('pasillo_options', 'value'),
              [Input('pasillo_selector', 'value')])
def display_status(selector):
    if selector == 'Custom':
        return ['Frutas']
    else:
        return []

# Radio -> multi
@app.callback(Output('categoria_options', 'value'),
              [Input('categoria_selector', 'value'),
               Input("pasillo_options","value")])
def display_status(pasillo_options,selector):
    if len(pasillo_options)>1:
        list = df_pasillo_cat[df_pasillo_cat.Pasillo.isin(pasillo_options)].Categoria.unique().tolist()
    else:
        list = df_pasillo_cat[df_pasillo_cat.Pasillo.str.contains(pasillo_options)].Categoria.unique().tolist()

    if selector == 'Custom':
        return list
    else:
        return []

@app.callback(
    Output('bogota-serie', 'figure'),
    [Input('none', 'children')])
def update_y_timeseries(value):
    df_sb_b = generate_ts(df, location="Bogotá")
    return {'data': [go.Scatter(x=df_sb_b.index,
                                y=df_sb_b.cantidadVendida.values,
                                line=dict(color = (app_colors['sentiment-plot']),width = 1,))],
            'layout': go.Layout(
                title='Cantidad vendida Bogotá',
                font={'color':app_colors['text']},
                plot_bgcolor = backgroundGraphs['background'],
                paper_bgcolor = backgroundGraphs['background'])}


@app.callback(
    dash.dependencies.Output('medellin-serie', 'figure'),
    [dash.dependencies.Input('none', 'children')])
def update_y_timeseries(value):
    df_sb_b = generate_ts(df, location="Medellín")
    return {'data': [go.Scatter(x=df_sb_b.index,
                                y=df_sb_b.cantidadVendida.values,
                                line=dict(color = (app_colors['sentiment-plot']),width = 1,))],
            'layout': go.Layout(
                title='Cantidad vendida Medellin',
                font={'color':app_colors['text']},
                plot_bgcolor = backgroundGraphs['background'],
                paper_bgcolor = backgroundGraphs['background'])}

if __name__ == '__main__':
    app.run_server(debug=True,
                   host="0.0.0.0")
