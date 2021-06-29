import plotly.graph_objects as go

import Calculations

# color_darkgreen ='#2ca02c'
# color_lightgreen ='#bcbd22'
# color_darkblue ='#1f77b4'
# color_lightblue ='#17becf'
# color_purple ='#9467bd'
# color_pink = '#e377c2'
# color_orange ='#ff7f0e'
# color_brown ='#8c564b'
# color_darkgrey ='#888888'
# color_lightgrey ='#a9a9a9'
# color_black ='#2f4f4f'

#############################################################################


def fig(input1, input2, input3, input4, N, lim):

    fig = Calculations.SpectrumDOSPlot(input1, input2, input3, input4, N, lim)



    # # Create figure
    # fig = go.Figure()

    # # Constants
    # img_width = 1600
    # img_height = 900
    # scale_factor = 0.5


    # # Add invisible scatter trace.
    # # This trace is added to help the autoresize logic work.
    # fig.add_trace(
    #     go.Scatter(
    #         x=[0, img_width * scale_factor],
    #         y=[0, img_height * scale_factor],
    #         mode="markers",
    #         marker_opacity=0
    #     )
    # )

    # # Configure axes
    # fig.update_xaxes(
    #     visible=False,
    #     range=[0, img_width * scale_factor]
    # )

    # fig.update_yaxes(
    #     visible=False,
    #     range=[0, img_height * scale_factor],
    #     # the scaleanchor attribute ensures that the aspect ratio stays constant
    #     scaleanchor="x"
    # )


    # fig.add_layout_image(
    # dict(
    #     x=0,
    #     sizex=img_width * scale_factor,
    #     y=img_height * scale_factor,
    #     sizey=img_height * scale_factor,
    #     xref="x",
    #     yref="y",
    #     opacity=1.0,
    #     layer="below",
    #     sizing="stretch",
    #     source=imagefilepath)
    # )


    # # Configure other layout
    # fig.update_layout(
    #     width=img_width * scale_factor,
    #     height=img_height * scale_factor,
    #     margin={"l": 0, "r": 0, "t": 0, "b": 0},
    # )

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/

    return fig