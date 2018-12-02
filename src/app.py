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
from LSTM_ts import *
from keras.models import load_model
from flask_caching import Cache
import dash_table as dt

# Setup the app
app = dash.Dash('MerqueoModelDashboard')

CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': "cache-directory"
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

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
                    html.P('Filter by product:'),
                    dcc.Dropdown(id='product_options',
                                 options=products_opt,
                                 value=products[0])
            ],className="six columns"),
            html.Div([
                html.P('Select train split:'),
                html.Div([dcc.Slider(id='train_test_split',
                                     min=0.4,
                                     max=1,
                                     step=0.1,
                                     marks={i: '{}'.format(np.round(i, 1)) for i in\
                                            np.linspace(start=0.4, stop=1, num=7)},
                                     value=0.7)],
                         style={'margin-top': '10',
                                'margin-right':'20',
                                'margin-left':'20',}),
            ], className="four columns", style={'margin-top': '5', 'align':"right"}),
        ], className="row", style={'margin-top': '30',}),
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
                    "Total Epochs",
                    "total_epoch",
                ),
            indicator(
                    "#EF553B",
                    "MAE",
                    "MAE_model",
                )], className="row"),
        html.Div([
            html.Div(
                [
                    dcc.Graph(id='cantidad_vendida_serie',
                              animate=False)
                ], className='six columns',
                style={'margin-top': '20'}),
            html.Div(
                [
                    dcc.Graph(id='individual_graph_1')
                ], className='four columns',
                style={'margin-top': '20'}),
            html.Div(
                [
                    dt.DataTable(id="forecast_table",
                                 columns=[{'id':'Date','name':'Date'},
                                          {'id':'Prediction','name':'Prediction'}],
                                 data=[{}],
                                 style_cell={
                                     # 'backgroundColor': 'rgb(50, 50, 50)',
                                     # 'color': 'white',
                                     'textAlign': 'center',
                                     'fontSize':18
                                 },
                                 style_as_list_view=True,
                                 style_header={
                                     # 'backgroundColor': 'rgb(30, 30, 30)',
                                     'fontWeight': 'bold',
                                     'fontSize':18})
                ], className='two columns',
                style={'margin-top': '20'})
            ], className="row")
        ], className="row", style={'margin-right':'20', 'margin-left':'20'}),
        html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet"),
        html.Link(href="https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css",rel="stylesheet"),
        html.Pre(id='output'),
        # hidden signal value
        html.Div(id='signal', style={'display': 'none'})
    ],
    style={'font' : dict(family = "Helvetica",
                       size = 14,
                       color = '#CCCCCC',)}
),style={'backgroundColor':'#ffff'})


# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize(timeout=appconfig.TIMEOUT)
def global_store(tools_):
    product_options = tools_[0]
    train_test_split = tools_[1]

    dff = subset_dataframe_2(df=df, product_options=product_options)
    df_sb_b = dff.resample('D').sum()

    ## Model
    # train the model
    serie = create_sequence(df_sb_b.cantidadVendida)
    data_model = train_test_reshape(serie=serie,
                                    product_name=product_options,
                                    split=train_test_split)
    history = train_model(data_trainig=data_model)
    model_loss = history.history['loss']
    model_val_loss = history.history['val_loss']

    # model predictions
    file_name = "../Output/model_%s.h5" % data_model['product_name']
    file_name = file_name
    error, prediction, train_prediction = compute_mae(model=load_model(file_name),
                                    data_trainig=data_model)
    forcast_7_days, new_index = forecasting_7_days(model=load_model("../Output/model_%s.h5"\
                                                                    % data_model['product_name']),
                                                   data_trainig=data_model)
    data_model.update({"model_loss": model_loss,
                       "model_val_loss": model_val_loss,
                       "error": [error],
                       "test_prediction": prediction,
                       "train_prediction": train_prediction,
                       "forcast_7_days": forcast_7_days,
                       "new_index": new_index})
    # # write the output
    # if os.path.exists("../Output/dict.pickle"):
    #     os.remove("../Output/dict.pickle")
    # else:
    #     print("The model does not exist")
    # pickle_out = open("../Output/dict.pickle", "wb")
    # pickle.dump(data_model, pickle_out)
    # pickle_out.close()

    return data_model


@app.callback(Output('signal', 'children'),
              [Input('product_options', 'value'),
               Input('train_test_split', 'value')])
def compute_value(value, value2):
    # compute value and send a signal when done
    value = [value, value2]
    global_store(value)
    return value


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
              [Input('product_options', 'value')])
def total_deliveries_indicator_callback(product_options):
    dff = subset_dataframe_2(df, product_options)
    return "{}K".format(str(np.round(dff.sum()["cuenta_gr"].sum()/appconfig.orders_scale, 1)))


@app.callback(Output('total_epoch', 'children'),
              [Input('product_options', 'value'),
               Input('train_test_split', 'value'),
               Input('signal', 'children')])
def prct_deliveries_indicator_callback(product_options, train_test_split, value):
    dict_G = global_store(value)
    return "{}".format(len(dict_G["model_loss"]))

@app.callback(Output('MAE_model', 'children'),
              [Input('product_options', 'value'),
               Input('signal', 'children')])
def prct_deliveries_indicator_callback(product_options, value):
    dict_G = global_store(value)
    return "%.2f" % (dict_G["error"][0])

# Graph real vs Predicted ts and forecasting for 7 days
@app.callback(Output('cantidad_vendida_serie', 'figure'),
              [Input('product_options', 'value'),
               Input('train_test_split', 'value'),
               Input('signal', 'children')])
def update_y_timeseries(product_options, train_test_split, value):

    dict_G = global_store(value)

    return {'data': [go.Scatter(x=dict_G["train_index"].tolist()+\
                                  dict_G["test_index"].tolist(),
                                y=dict_G["serie"].values,
                                name="real"),
                     go.Scatter(x=dict_G["train_index"].tolist(),
                                y=dict_G["train_prediction"].flatten(),
                                name="Prediction\nTrain"),
                     go.Scatter(x=dict_G["test_index"].tolist(),
                                y=dict_G["test_prediction"].flatten(),
                                name="Prediction\nTest"),
                     go.Scatter(x=dict_G["new_index"],
                                y=dict_G["forcast_7_days"],
                                name="Forecast(7 days)")],

            'layout': go.Layout(
                title='Daily Sold Quantities %s' % product_options,
                plot_bgcolor=backgroundGraphs['background'],
                paper_bgcolor=backgroundGraphs['background'],
                xaxis=dict(title='Time in Days',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                yaxis=dict(title='Sold Quantities',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                font = dict(color='#e8e9e9', size=14)
            )}

# Graph for learning curve
@app.callback(Output('individual_graph_1', 'figure'),
              [Input('product_options', 'value'),
               Input('signal', 'children')])
def update_y_timeseries(product_options,value):

    dict_G = global_store(value)

    return {'data': [go.Scatter(x=list(range(len(dict_G["model_loss"]))),
                                y=dict_G["model_loss"],
                                name="Training"),
                     go.Scatter(x=list(range(len(dict_G["model_loss"]))),
                                y=dict_G["model_val_loss"],
                                name="Validation")],
            'layout': go.Layout(
                title='Learning Curve',
                plot_bgcolor=backgroundGraphs['background'],
                paper_bgcolor=backgroundGraphs['background'],
                xaxis=dict(title='Epoch',
                           zerolinecolor='#909395'),
                yaxis=dict(title='Mean Absolute Error',
                           gridcolor='#4e5256',
                           zerolinecolor='#909395'),
                font=dict(color='#e8e9e9', size=14)

            )}
# table
@app.callback(Output('forecast_table','data'),
              [Input('product_options', 'value'),
               Input('signal', 'children')])
def make_table_forecast(product_options,value):
    dict_G = global_store(value)
    index_ = [date.strftime('%Y-%m-%d') for date in dict_G["new_index"]]
    predictions_ = [round(num_) for num_ in dict_G["forcast_7_days"]]
    dff = pd.DataFrame(list(zip(index_,predictions_)), columns=['Date', 'Prediction'])
    return dff.to_dict("rows")



if __name__ == '__main__':
    app.run_server(debug=True,
                   host="0.0.0.0",
                   threaded=True)
