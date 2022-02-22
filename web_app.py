import datetime
from datetime import datetime as dt
import time

import numpy as np

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from flask import Flask
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import dash_cytoscape as cyto


if __name__ == "__main__":
    print("start web app")
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)

    title_div = html.Div(html.H1("CLIP Semantic Search"), className="title")

    # add text input
    text_input = dcc.Input(id="input-box", type="text", placeholder="Enter a query")

    app.layout = html.Div(children=[title_div, text_input])

    # @app.callback(Output('cytoscape-tapNodeData-json', 'children'),
    #               [Input('cytoscape-event-callbacks-1', 'tapNodeData')])
    # def displayTapNodeData(data):
    #     return json.dumps(data, indent=2)
    #
    #
    # @app.callback(
    #     Output('model_data', 'children'),
    #     [Input("btn_add_node", "n_clicks_timestamp")]
    # )
    # def callback(value):
    #     return "pressed"
    #     print("pressed")

    # @app.callback(
    #     Output('r0_slider_text', 'children'),
    #     [Input("r0_slider", "value")]
    # )
    # def callback(value):
    #     return r0_callback(value)
    #
    #
    # @app.callback(
    #     Output('seasonality_slider_text', 'children'),
    #     [Input("seasonality_slider", "value")]
    # )
    # def callback(value):
    #     return seasonality_callback(value)
    #
    #
    # # duration callbacks
    # @app.callback(
    #     Output('latency_time_slider_text', 'children'),
    #     [Input("latency_time_slider", "value")]
    # )
    # def callback(value):
    #     return latency_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('infectious_time_slider_text', 'children'),
    #     [Input("infectious_time_slider", "value")]
    # )
    # def callback(value):
    #     return infectious_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('incubation_time_slider_text', 'children'),
    #     [Input("incubation_time_slider", "value")]
    # )
    # def callback(value):
    #     return incubation_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('symptom_time_slider_text', 'children'),
    #     [Input("symptom_time_slider", "value")]
    # )
    # def callback(value):
    #     return symptom_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('symptom2hospital_time_slider_text', 'children'),
    #     [Input("symptom2hospital_time_slider", "value")]
    # )
    # def callback(value):
    #     return symptom2hospital_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('hospital_time_slider_text', 'children'),
    #     [Input("hospital_time_slider", "value")]
    # )
    # def callback(value):
    #     return hospital_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('hospital2intensive_time_slider_text', 'children'),
    #     [Input("hospital2intensive_time_slider", "value")]
    # )
    # def callback(value):
    #     return hospital2intensive_time_callback(value)
    #
    #
    # @app.callback(
    #     Output('intensive_time_slider_text', 'children'),
    #     [Input("intensive_time_slider", "value")]
    # )
    # def callback(value):
    #     return intensive_time_callback(value)
    #
    #
    # # probability callbacks
    # @app.callback(
    #     Output('p_hospital_slider_text', 'children'),
    #     [Input("p_hospital_slider", "value")]
    # )
    # def callback(value):
    #     return p_hospital_callback(value)
    #
    #
    # @app.callback(
    #     Output('p_hospital_intensive_slider_text', 'children'),
    #     [Input("p_hospital_intensive_slider", "value")]
    # )
    # def callback(value):
    #     return p_hospital_intensive_callback(value)
    #
    #
    # @app.callback(
    #     Output('p_intensive_dead_slider_text', 'children'),
    #     [Input("p_intensive_dead_slider", "value")]
    # )
    # def callback(value):
    #     return p_intensive_dead_callback(value)
    #
    #
    # @app.callback(
    #     Output("Graph", "figure"),
    #     [Input("checklist", "value"),
    #      Input("r0_slider", "value"),
    #      Input("seasonality_slider", "value"),
    #      Input("p_hospital_slider", "value"),
    #      Input("p_hospital_intensive_slider", "value"),
    #      Input("p_intensive_dead_slider", "value"),
    #
    #      Input("latency_time_slider", "value"),
    #      Input("infectious_time_slider", "value"),
    #      Input("incubation_time_slider", "value"),
    #      Input("symptom_time_slider", "value"),
    #      Input("symptom2hospital_time_slider", "value"),
    #      Input("hospital_time_slider", "value"),
    #      Input("hospital2intensive_time_slider", "value"),
    #      Input("intensive_time_slider", "value")]
    # )
    # def select_graphs(selection, r0, seasonality, p_hospital, p_hospital_intensive, p_intensive_dead,
    #                   latency_time, infectious_time, incubation_time, symptom_time, symptom2hospital_time,
    #                   hospital_time, hospital2intensive_time, intensive_time):
    #     return calculate_graph(selection, r0, seasonality, p_hospital, p_hospital_intensive, p_intensive_dead,
    #                            latency_time, infectious_time, incubation_time, symptom_time, symptom2hospital_time,
    #                            hospital_time, hospital2intensive_time, intensive_time, SYMPTOM2TEST_TIME)

    app.run_server()
