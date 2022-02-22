import datetime
from datetime import datetime as dt
import time

import numpy as np

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image, ImageOps
import numpy as np
import plotly.express as px

from search import search

# import dash_cytoscape as cyto


if __name__ == "__main__":
    print("start web app")
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)

    title_div = html.Div(html.H1("CLIP Semantic Search"), className="title")

    # add text input
    text_input = dcc.Input(id="input-box", type="text", placeholder="Enter a query")

    # add label
    label1 = html.Label("Enter a query", id="label1")

    # add button
    button = html.Button(id="button", children="Submit")

    #top_img_files, top_values = search("dog", n_results=6)

    content_div = html.Div(id="content-div")







    app.layout = html.Div(children=[title_div, text_input, label1, button, content_div])


    # @app.callback(Output("output", "children"), Input("input", "value"))
    # def update_output(input):
    #     if not input:
    #         raise PreventUpdate
    #
    #     try:
    #         img = np.array(Image.open("test.jpg"))
    #     except OSError:
    #         raise PreventUpdate
    #
    #     fig = px.imshow(img, color_continuous_scale="gray")
    #     fig.update_layout(coloraxis_showscale=False)
    #     fig.update_xaxes(showticklabels=False)
    #     fig.update_yaxes(showticklabels=False)
    #
    #     return dcc.Graph(figure=fig)

    @app.callback(
        Output(component_id="content-div", component_property="children"),
        [Input(component_id="button", component_property="n_clicks")],
        [State(component_id="input-box", component_property="value")]
    )
    def update_label(n_clicks, input_value):
        print(n_clicks)
        if n_clicks is None:
            print("n_clicks is None")
            return "Enter a query"
        else:
            print(input_value)
            top_img_files, top_values = search(input_value, n_results=6)
            print("n_clicks is not None")
            graphs = []

            for i, img_file in enumerate(top_img_files):
                img = Image.open(img_file)
                img = ImageOps.exif_transpose(img)
                img = np.array(img)

                # img = np.array(Image.open("test.JPG"))
                fig = px.imshow(img, color_continuous_scale="gray")
                fig.update_layout(coloraxis_showscale=False)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                g = dcc.Graph(figure=fig, config={'staticPlot': True})  # , style={'width': '30vh', 'height': '30vh'})
                graphs.append(g)
            return graphs




    app.run_server(debug=True, port = 8080, host ="0.0.0.0")
    #app.run_server(debug=True, port=8080, host="192.168.178.55")
    #app.run_server()
