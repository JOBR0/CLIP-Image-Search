import base64
import time

import clip

import dash
import flask
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
from flask import Flask

from PIL import Image, ImageOps
import numpy as np
import plotly.express as px

from search import search, load_features, encode_text

## Diskcache
import diskcache

from functools import partial


def init_app():

    print("start web app")
    #server = Flask(__name__)
    cache = diskcache.Cache("./cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)
    #app = dash.Dash(__name__, server=server, long_callback_manager=long_callback_manager)
    app = dash.Dash(__name__, long_callback_manager=long_callback_manager)


    device = "cpu"

    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_features, paths = load_features(device, "C:/Users/Jonas/Desktop/reps.csv")

    title_div = html.Div(html.H1("CLIP Semantic Search"), className="title")

    # add text input
    text_input = dcc.Input(id="input-box", type="text", placeholder="Enter a query")

    # add label
    label1 = html.Label("Enter a query", id="label1")

    # add button
    button = html.Button(id="button", children="Submit")

    # top_img_files, top_values = search("dog", n_results=6)

    content_div = html.Div(id="content-div")

    image_filename = "test2.png"  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())

    static_image_route = "/static/"

    #img1 = html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    #img1 = html.Img(src=static_image_route + "test2.png")
    #img2 = html.Img(src=static_image_route + "test.JPG")


    store = dcc.Store(id="image-paths")

    app.layout = html.Div(children=[title_div, text_input, label1, button, content_div, store])

    # Add a static image route that serves images from desktop
    # Be *very* careful here - you don't want to serve arbitrary files
    # from your computer or server
    @app.server.route(f"{static_image_route}<image_path>")
    def serve_image(image_path):
        #image_name = '{}.png'.format(image_path)
        # if image_name not in list_of_images:
        #     raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
        return flask.send_from_directory("", image_path)

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

    # @app.long_callback(
    #     output=Output("content-div", "children"),
    #     inputs=Input("button", "n_clicks"),
    #     state=State(component_id="input-box", component_property="value"),
    #     running=[(Output("button", "disabled"), True, False)],
    #     progress=Output("label1", "children"),
    #     manager=long_callback_manager,
    # )
    def callback(set_progress, n_clicks, input_value, model=model, device=device, image_features=image_features, paths=paths):
        print("callback")
        print(n_clicks)
        print(input_value)
        imgs = []
        for i in range(10):
            img = html.Img(src=static_image_route + "test.JPG", style={"height": "500px"})
            # img = Image.open("test.JPG")
            # img = ImageOps.exif_transpose(img)
            # img = np.array(img)

            # # img = np.array(Image.open("test.JPG"))
            # fig = px.imshow(img, color_continuous_scale="gray")
            # fig.update_layout(coloraxis_showscale=False)
            # fig.update_xaxes(showticklabels=False)
            # fig.update_yaxes(showticklabels=False)
            # g = dcc.Graph(figure=fig, config={'staticPlot': True})  # , style={'width': '30vh', 'height': '30vh'})
            #g = "test"
            imgs.append(img)
            # set_progress(f"{i}/{len(top_img_files)}")
            #            set_progress(graphs)
            # children.append("{}".format(i))
            # time.sleep(1)
            set_progress(f"{i}/{len(imgs)}")

        return imgs

    @app.long_callback(
        output=Output("content-div", "children"),
        inputs=Input("button", "n_clicks"),
        state=State(component_id="input-box", component_property="value"),
        running=[(Output("button", "disabled"), True, False)],
        progress=Output("label1", "children"),
        manager=long_callback_manager,)
    def callback_wrapper(set_progress, n_clicks, input_value):
        return callback(set_progress, n_clicks, input_value)

    # @app.long_callback(
    #     Output(component_id="image-paths", component_property="data"),
    #     Input(component_id="button", component_property="n_clicks"),
    #     State(component_id="input-box", component_property="value"),
    #     # running=[
    #     #     (Output("button", "disabled"), True, False)
    #     # ],
    #     # cancel=[Input("cancel_button_id", "n_clicks")],
    #     # progress=Output("content-div", "children"),
    #     progress=Output("label1", "children"),
    #     # progress_default=[],
    #     # interval=1000,
    # )
    # def callback(set_progress, n_clicks, input_value, model=model, device=device, image_features=image_features,
    #              paths=paths):
    #     print(n_clicks)
    #     if n_clicks is None:
    #         print("n_clicks is None")
    #         return None
    #     else:
    #         print(input_value)
    #         text_features = encode_text(input_value, model, device)
    #
    #         print("start search")
    #         top_img_files, top_values = search(query_features=text_features,
    #                                            image_features=image_features,
    #                                            paths=paths,
    #                                            n_results=6)
    #         print("end search")
    #         print("n_clicks is not None")
    #         graphs = []
    #
    #         for i, img_file in enumerate(top_img_files):
    #             img = Image.open(img_file)
    #             img = ImageOps.exif_transpose(img)
    #             img = np.array(img)
    #
    #             # img = np.array(Image.open("test.JPG"))
    #             fig = px.imshow(img, color_continuous_scale="gray")
    #             fig.update_layout(coloraxis_showscale=False)
    #             fig.update_xaxes(showticklabels=False)
    #             fig.update_yaxes(showticklabels=False)
    #             g = dcc.Graph(figure=fig, config={'staticPlot': True})  # , style={'width': '30vh', 'height': '30vh'})
    #             graphs.append(g)
    #             set_progress(f"{i}/{len(top_img_files)}")
    #             # set_progress(graphs)
    #         print("return graphs")
    #         return None

    # @app.callback(
    #     Output(component_id="content-div", component_property="children"),
    #     [Input(component_id="button", component_property="n_clicks")],
    #     [State(component_id="input-box", component_property="value")]
    # )
    # def update_label(n_clicks, input_value):
    #     print(n_clicks)
    #     if n_clicks is None:
    #         print("n_clicks is None")
    #         return "Enter a query"
    #     else:
    #         print(input_value)
    #         text_features = encode_text(input_value, model, device)
    #
    #         print("start search")
    #         top_img_files, top_values = search(query_features=text_features,
    #                                            image_features=image_features,
    #                                            paths=paths,
    #                                            n_results=6)
    #         print("end search")
    #         print("n_clicks is not None")
    #         graphs = []
    #
    #         for i, img_file in enumerate(top_img_files):
    #             img = Image.open(img_file)
    #             img = ImageOps.exif_transpose(img)
    #             img = np.array(img)
    #
    #             # img = np.array(Image.open("test.JPG"))
    #             fig = px.imshow(img, color_continuous_scale="gray")
    #             fig.update_layout(coloraxis_showscale=False)
    #             fig.update_xaxes(showticklabels=False)
    #             fig.update_yaxes(showticklabels=False)
    #             g = dcc.Graph(figure=fig, config={'staticPlot': True})  # , style={'width': '30vh', 'height': '30vh'})
    #             graphs.append(g)
    #         print("return graphs")
    #         return graphs


    # app.run_server(debug=True, port=8080, host="192.168.178.55")
    # app.run_server()
    return app


if __name__ == "__main__":
    app = init_app()
    # use_reloader=False prevents wierd multiple runs
    app.run_server(debug=True, port=8080, host="0.0.0.0", use_reloader=False)
