import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from __main__ import *
import Calculations
import Controls
import Callbacks

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

##################################################################
##################################################################
# LAYOUT

app.layout = dbc.Container(
    [
        html.H1(children='Twisted Graphene'),
        html.Hr(),

        dcc.Markdown(children=""" Calculate and visualize the bandstructure and local density of states of 
        twisted graphene systems with the parameters of your choice."""),

        dbc.Row([
            dbc.Col(Controls.Card, md=4),
            dbc.Col(dcc.Graph(id="RawDataGraph"), md=8),
        ], align="center",),

        html.Hr(),

        dcc.Markdown(children="""Questions? [Send me an e-mail](mailto:leo.goutte@mail.mcgill.ca)
        
        Made by: Leo Goutte
        For: Condensed matter physiscists in a hurry 
        """),
        html.Hr(),

    ],
    fluid=True,
)
##################################################################
##################################################################
# CALLBACKS

@app.callback(
    Output('RawDataGraph', 'figure'),
    [Input('Input1value', 'value'),
     Input('Input2value', 'value'),
     Input('Input3value', 'value'),
     Input('Input4value', 'value'),
     Input('Nvalue', 'value'),
     Input('limvalue', 'value')])
def update_figure(input1, input2, input3, input4, N, lim):
    fig = Callbacks.fig(input1, input2, input3, input4, N, lim)
    return fig


#############################################################################
# RUN ON SERVER
if __name__ == '__main__':
    app.run_server(debug=True)
