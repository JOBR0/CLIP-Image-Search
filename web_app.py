import base64
import datetime
import io
import os
import time

import clip

import dash
import flask
import torch
from dash import dcc, ALL
from dash import html
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager
from flask import Flask

from PIL import Image, ImageOps
import numpy as np
import plotly.express as px

from search import search, load_features, encode_text, encode_images

## Diskcache
import diskcache

from functools import partial

N_RESULTS = 30
STATIC_IMAGE_ROUTE = "/static/"


def create_img_div(img_path):
    folder = os.path.basename(os.path.dirname(os.path.normpath(img_path)))
    # folder = os.path.basename(os.path.normpath(img_path))

    outer_box = html.Div(
        style={"color": "white", "background-color": "black", "padding": "5px", "margin": "5px", "float": "left"})

    title_box = html.Div(children=folder, style={"background-color": "green", "text-align": "center", "padding": "5px"})

    img = html.Img(src=STATIC_IMAGE_ROUTE + img_path, style={"height": "200px"}, title=img_path)

    outer_box.children = [title_box, img]

    return outer_box


def b64_string_to_pil(base64_string):
    # remove header
    base64_string = base64_string.split(",")[1]

    imgdata = base64.b64decode(str(base64_string))
    return Image.open(io.BytesIO(imgdata))


def parse_contents(contents, filename, date, index):
    # extract image data and decode
    # pil_img = b64_string_to_pil(contents.split(",")[1])

    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={"height": "150px"}),
        html.Button(id={"type": "del-button", "index": index}, children="X",
                    style={"position": "absolute", "top": "-5px", "right": "-5px"}),
    ]
        , style={"position": "relative", "float": "left", "margin": "10px"})


def init_app():
    print("start web app")
    # server = Flask(__name__)
    cache = diskcache.Cache("./cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)
    # app = dash.Dash(__name__, server=server, long_callback_manager=long_callback_manager)
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__,
                    long_callback_manager=long_callback_manager)  # , external_stylesheets=external_stylesheets)

    app.config.suppress_callback_exceptions = True

    device = "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_features = np.load("image_features.npy")

    # add text input
    text_input = dcc.Input(id="input-box", type="text", placeholder="Enter a Text Query",
                           style={"font-size": "50px", "width": "100%", "box-sizing": "border-box"})

    # add button
    button = html.Button(id="button", children="Search", style={"font-size": "50px"})

    ctrl_div_left = html.Div(children=[text_input, button],
                             style={"float": "left", "background-color": "blue", "width": "50%",

                                    "padding": "10px", "box-sizing": "border-box"})

    empty_div = html.Div([
        'Drag and Drop or ',
        html.A('Select Image Queries'),

    ],
        id='empty-div', style={"text-align": "center", "width": "100%"})

    image_query_div = html.Div(id='output-image-upload')

    upload = dcc.Upload(
        id='upload-image',
        children=
        [empty_div, image_query_div],
        style={
            'width': '100%',
            "float": "right",
            "box-sizing": "border-box",
            #'height': '300px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            # 'margin': '10px',
            'background-color': 'green'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )

    # upload_output = html.Div(id='output-image-upload', style={"float": "left"})

    # store = dcc.Store(id="image-paths")
    ctrl_div_right = html.Div(children=[upload], style={"float": "right", "background-color": "red", "width": "50%",
                                                        "padding": "10px", "box-sizing": "border-box"})

    ctrl_div = html.Div(children=[ctrl_div_left, ctrl_div_right],
                        style={"padding": "0px", "background-color": "pink" })

    content_div = html.Div(id="content-div", style={ "background-color": "orange", "display":"inline-block"})

    app.layout = html.Div(children=[ctrl_div, content_div])

    app.title = "CLIP Search"

    # @app.callback(Output('output-image-upload', 'children'),
    #               Input('del-button', 'n_clicks'),
    #               State('output-image-upload', 'children'))
    # def update_output(n_clicks, current_children):
    #     children = current_children or []
    #
    #     print("event")
    #
    #     # if list_of_contents is not None:
    #     #     children = current_children + [
    #     #         parse_contents(c, n, d) for c, n, d in
    #     #         zip(list_of_contents, list_of_names, list_of_dates)]
    #
    #     if len(children) > 0:
    #         empty_div_style = {'display': 'none'}
    #     else:
    #         empty_div_style = {'display': 'block'}
    #
    #     return []

    @app.callback(Output('output-image-upload', 'children'),
                  Output('empty-div', 'style'),
                  Output("upload-image", "disable_click"),
                  Input({'type': 'del-button', 'index': ALL}, 'n_clicks'),
                  Input('upload-image', 'contents'),
                  State('upload-image', 'filename'),
                  State('upload-image', 'last_modified'),
                  State('output-image-upload', 'children'))
    def update_output(n_clicks, list_of_contents, list_of_names, list_of_dates, current_children):
        current_children = current_children or []

        clicked = False

        if n_clicks is not None:
            children = []
            for i, clicks in enumerate(n_clicks):
                if clicks is None:
                    children.append(current_children[i])
                else:
                    clicked = True
        else:
            children = current_children

        n_current_children = len(children)

        if list_of_contents is not None and not clicked:
            children = children + [
                parse_contents(c, n, d, n_current_children) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]

        if len(children) > 0:
            empty_div_style = {'display': 'none'}
            disable_click = True
        else:
            empty_div_style = {'display': 'block'}
            disable_click = False

        return children, empty_div_style, disable_click

    # Add a static image route that serves images from desktop
    # Be *very* careful here - you don't want to serve arbitrary files
    # from your computer or server
    # @app.server.route(f"{static_image_route}<image_path>")
    @app.server.route(f"{STATIC_IMAGE_ROUTE}<path:filename>.<ext>")
    def serve_image(filename, ext):
        print(f"serving {filename}.{ext}")
        # if image_name not in list_of_images:
        #     raise Exception('"{}" is excluded from the allowed static files'.format(image_path))

        return flask.send_from_directory("//nas_enbaer", f"{filename}.{ext}")
        # return flask.send_from_directory("", image_path)

    @app.callback(
        output=Output("content-div", "children"),
        inputs=Input("button", "n_clicks"),
        state=[State(component_id="input-box", component_property="value"),
               State(component_id="output-image-upload", component_property="children")]
        , )
    # running=[(Output("button", "disabled"), True, False)],
    # progress=Output("label1", "children"),
    # manager=long_callback_manager,)
    def callback(n_clicks, text_input, image_inputs):  # , model=model, device=device, image_features=image_features):
        print("callback")
        print(n_clicks)
        if n_clicks is None:
            return []
        print(text_input)

        if text_input is not None and text_input.strip() != "":
            text_query = encode_text(text_input, model, device)
        else:
            text_query = torch.zeros(0, 512)

        if image_inputs is not None and len(image_inputs) > 0:
            # Extract images from children
            b64 = [inp["props"]["children"][0]["props"]["src"] for inp in image_inputs]
            images = [b64_string_to_pil(b64_img) for b64_img in b64]
            image_queries = encode_images(images, model, preprocess, device)
        else:
            image_queries = torch.zeros(0, 512)

        query_features = torch.cat((text_query, image_queries), dim=0)

        if query_features.shape[0] == 0:
            print("No queries")
            return ['Drag and Drop or ',
                    html.A('Select Image Queries'), ]
        elif query_features.shape[0] > 1:
            query_features = query_features.mean(dim=0).unsqueeze(0)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)

        print("start search")
        top_img_files, top_values = search(query_features=query_features.cpu().numpy(),
                                           image_features=image_features,
                                           n_results=N_RESULTS)
        print("end search")
        imgs = []
        for i, img_file in enumerate(top_img_files):
            path = img_file[13:]

            img = create_img_div(path)

            # img = html.Img(src=STATIC_IMAGE_ROUTE + path, style={"height": "500px"})
            imgs.append(img)
            # set_progress(f"{i}/{len(top_img_files)}")

        return imgs

    # @app.long_callback(
    #     output=Output("content-div", "children"),
    #     inputs=Input("button", "n_clicks"),
    #     state=State(component_id="input-box", component_property="value"),
    #     running=[(Output("button", "disabled"), True, False)],
    #     progress=Output("label1", "children"),
    #     manager=long_callback_manager,)
    # def callback_wrapper(set_progress, n_clicks, input_value):
    #     return callback(set_progress, n_clicks, input_value)

    return app


if __name__ == "__main__":
    app = init_app()
    # use_reloader=False prevents wierd multiple runs
    app.run_server(debug=True, port=8080, host="0.0.0.0", use_reloader=False)
