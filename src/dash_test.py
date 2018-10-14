# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html



# Setup the app
app = dash.Dash('MerqueoModelDashboard')

#Load the css
for source in ["https://codepen.io/jeanmidevacc/pen/paxKzB.css","https://codepen.io/pixinema/pen/XZvJyX.css"]:
    app.css.append_css({"external_url": source})

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True,host="0.0.0.0")
