import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

###########################################################

Card = dbc.Card([
    # dbc.CardHeader("Parameters"),
    html.H4("Input parameters", className="card-title"),
    dbc.FormGroup([
        dbc.Row([
            dbc.Col(dbc.Label("θ (°)", id="Input1Label"), md=8)], justify="between"),
        # un-comment for slider functionality
        # dbc.Row([
        #     dbc.Col(html.Div([
        #         dcc.Slider(
        #             id="Input1value",
        #             min=0,
        #             max=2,
        #             step=0.05,
        #             value=1,
        #             marks={
        #                 0:'0',
        #                 0.5:'0.5',
        #                 1:'1',
        #                 1.05:'*',
        #                 1.5:'1.5',
        #                 2:'2'
        #             })
        #         ])),
        # ]),
        # html.Div(id="slider-output-container")
        dbc.Row([
            dbc.Col(dcc.Input(id='Input1value', type='number', step=0.01, value=1.05), md=12),
        ]),
    ]),

    dbc.FormGroup([
        dbc.Row([
            dbc.Col(dbc.Label("AA tunneling (meV)", id="Input2Label"), md=8)], justify="between"),
        dbc.Row([
            dbc.Col(dcc.Input(id='Input2value', type='number', value=110.7), md=12),
        ]),
    ]),

    dbc.FormGroup([
        dbc.Row([
            dbc.Col(dbc.Label("AB tunneling (meV)", id="Input3Label"), md=8)], justify="between"),
        dbc.Row([
            dbc.Col(dcc.Input(id='Input3value', type='number', value=110.7), md=12),
        ]),
    ]),

    dbc.FormGroup([
        dbc.Row([
            dbc.Col(dbc.Label("Valley (±1, K/K')", id="Input4Label"), md=8)], justify="between"),
        dbc.Row([
            dbc.Col(dcc.Input(id='Input4value', type='number', value=+1), md=12),
        ]),
    ]),

    dbc.FormGroup([
        dbc.Row([
            dbc.Col(dbc.Label("Truncation range", id="NLabel"), md=8)], justify="between"),
        dbc.Row([
            dbc.Col(dcc.Input(id='Nvalue', type='number', step=1, value=3), md=12),
        ]),
    ]),

    dbc.FormGroup([
        dbc.Row([
            dbc.Col(dbc.Label("Energy limit (meV)", id="limLabel"), md=8)], justify="between"),
        dbc.Row([
            dbc.Col(dcc.Input(id='limvalue', type='number', step=10, value=330), md=12),
        ]),
    ]),

], body=True)
