import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Output
from dash.dependencies import Input
import appConfig as appconfig


def subset_dataframe(df, location, filter_date, sections_selector, section_options):
    if location == 'All':
        location = ''
    if sections_selector == 'All':
        sections_selector = ''
        df_sb = df[marks_slider[filter_date[0]]:marks_slider[filter_date[1]]]
        df_sb = df_sb[df_sb.Ciudad.str.contains(location) &
                      df_sb.Pasillo.str.contains(sections_selector)]
    elif sections_selector == 'Custom':
        df_sb = df[marks_slider[filter_date[0]]:marks_slider[filter_date[1]]]
        df_sb = df_sb[df_sb.Ciudad.str.contains(location) &
                      df_sb.Pasillo.isin(section_options)]
    else:
        print("I'm doing something stupid")
    return df_sb

def generate_ts(df):
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