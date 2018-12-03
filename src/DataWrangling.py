import dash_html_components as html
import pandas as pd
import numpy as np

def subset_dataframe(df, location,
                     filter_date,
                     sections_selector,
                     section_options,
                     product_options):
    if location == 'All':
        location = ''

    if sections_selector == '':
        df_sb = df[marks_slider[filter_date[0]]:marks_slider[filter_date[1]]]
        df_sb = df_sb[df_sb.Ciudad.str.contains(location) &
                      df_sb.Pasillo.str.contains(sections_selector) &
                      df_sb.NombreProducto.str.contains(product_options)]
        return df_sb
    elif sections_selector == 'Custom':

        df_sb = df[marks_slider[filter_date[0]]:marks_slider[filter_date[1]]]
        df_sb = df_sb[df_sb.Ciudad.str.contains(location) &
                      df_sb.Pasillo.isin(section_options)]
        return df_sb
    else:
        print("I'm doing something stupid")

def subset_dataframe_2(df, product_options):
    df_sb = df[df.NombreProducto.str.contains(product_options)]
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

with open("products_filter.txt","r",encoding='utf-8') as f:
    products = f.read().split("\n")

df['FechaCreaciónOrden'] = pd.to_datetime(df.FechaCreaciónOrden)
df.set_index("FechaCreaciónOrden", inplace=True, drop=False)
df['month_year'] = df.FechaCreaciónOrden.apply(lambda x: "%d-%d" % (x.year, x.month))
df = df[(df.NombreProducto.isin(products)) &
        (df.cantidadVendida<=np.percentile(df.cantidadVendida, q=99.95))]

df = df["2017-07":]


products_opt = [{'label': str(pro),
                        'value': str(pro)}
                       for pro in sorted(products)]

ciudad_options = [{'label': str(city),
                        'value': str(city)}
                       for city in ["All"]+sorted(df.Ciudad.unique().tolist()+["All"])]

categoria_options = [{'label': str(categoria), 'value': str(categoria)}\
                      for categoria in ["All"]+sorted(df.Categoria.unique().tolist())]

pasillo_options = [{'label': str(pasillo), 'value': str(pasillo)}\
                      for pasillo in ["All"]+sorted(df.Pasillo.unique().tolist())]

marks_slider = dict(enumerate(df['month_year'].unique()))

