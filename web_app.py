import base64
import datetime
import io
import os
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

N_RESULTS = 30
STATIC_IMAGE_ROUTE = "/static/"

def create_img_div(img_path):
    folder = os.path.basename(os.path.dirname(os.path.normpath(img_path)))
    #folder = os.path.basename(os.path.normpath(img_path))

    outer_box = html.Div(style={"color": "white", "background-color": "black", "padding": "5px", "margin": "5px", "float": "left"})

    title_box = html.Div(children=folder, style={"background-color": "green", "text-align": "center", "padding": "5px"})

    img = html.Img(src=STATIC_IMAGE_ROUTE + img_path, style={"height": "200px"}, title=img_path)

    outer_box.children = [title_box, img]
    
    return outer_box

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    return Image.open(io.BytesIO(imgdata))

def parse_contents(contents, filename, date):
    # extract image data and decode
    pil_img = stringToRGB(contents.split(",")[1])

    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={"height": "50px"}),
        #html.Hr(),
        #html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])


def init_app():

    print("start web app")
    #server = Flask(__name__)
    cache = diskcache.Cache("./cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)
    #app = dash.Dash(__name__, server=server, long_callback_manager=long_callback_manager)
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, long_callback_manager=long_callback_manager)#, external_stylesheets=external_stylesheets)


    device = "cpu"

    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    #image_features, paths = load_features(device, "C:/Users/Jonas/Desktop/encoded_fixed.csv")

    image_features = np.load("image_features.npy")

    #title_div = html.Div(html.H1("CLIP Semantic Search"), className="title")

    # add text input
    text_input = dcc.Input(id="input-box", type="text", placeholder="Enter a query")

    # add label
    #label1 = html.Label("Enter a query", id="label1")

    # add button
    button = html.Button(id="button", children="Submit")

    # top_img_files, top_values = search("dog", n_results=6)

    content_div = html.Div(id="content-div")

    

    upload = dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )

    upload_output = html.Div(id='output-image-upload')




    store = dcc.Store(id="image-paths")

    app.layout = html.Div(children=[text_input, button, upload, upload_output, content_div, store])

    app.title = "CLIP Search"

    @app.callback(Output('output-image-upload', 'children'),
                  Input('upload-image', 'contents'),
                  State('upload-image', 'filename'),
                  State('upload-image', 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children

    # Add a static image route that serves images from desktop
    # Be *very* careful here - you don't want to serve arbitrary files
    # from your computer or server
    #@app.server.route(f"{static_image_route}<image_path>")
    @app.server.route(f"{STATIC_IMAGE_ROUTE}<path:filename>.<ext>")
    def serve_image(filename, ext):
        print(f"serving {filename}.{ext}")
        # if image_name not in list_of_images:
        #     raise Exception('"{}" is excluded from the allowed static files'.format(image_path))

        return flask.send_from_directory("//nas_enbaer", f"{filename}.{ext}")
        #return flask.send_from_directory("", image_path)

    @app.callback(
        output=Output("content-div", "children"),
        inputs=Input("button", "n_clicks"),
        state=State(component_id="input-box", component_property="value"),)
        #running=[(Output("button", "disabled"), True, False)],
        #progress=Output("label1", "children"),
        #manager=long_callback_manager,)
    def callback(n_clicks, input_value):#, model=model, device=device, image_features=image_features):
        print("callback")
        print(n_clicks)
        if n_clicks is None:
            return []
        print(input_value)
        print(input_value)
        text_features = encode_text(input_value, model, device).cpu().numpy()

        print("start search")
        top_img_files, top_values = search(query_features=text_features,
                                           image_features=image_features,
                                           n_results=N_RESULTS)
        print("end search")
        imgs = []
        for i, img_file in enumerate(top_img_files):
            path = img_file[13:]

            img = create_img_div(path)

            #img = html.Img(src=STATIC_IMAGE_ROUTE + path, style={"height": "500px"})
            imgs.append(img)
            #set_progress(f"{i}/{len(top_img_files)}")

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
