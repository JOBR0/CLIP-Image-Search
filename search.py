import dbm
import os

import numpy as np
import pandas as pd
from ast import literal_eval

import clip
import torch

import math

import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import time

def encode_images(image_files, model, preprocess, device):
    images = []
    for img in image_files:
        images.append(preprocess(img))

    images = torch.stack(images)

    with torch.inference_mode():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def encode_text(text, model, device):
    text = clip.tokenize(text).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def load_features(device, csv_file):
    df = pd.read_csv(csv_file)
    image_features = [literal_eval(f) for f in df["features"]]
    image_features = torch.tensor(image_features).to(device)
    paths = df["path"].to_numpy()
    return image_features, paths


def search(image_features, query_features, n_results=6):
    # cosine similarity as logits
    logits_per_image = image_features @ query_features.T

    logits_per_image = logits_per_image.squeeze()

    top_indices = np.argsort(logits_per_image, axis=0)[::-1][:n_results]

    top_values = logits_per_image[top_indices]

    top_img_files = []
    with dbm.open(os.path.join(".", "database"), 'r') as db:
        for idx in top_indices:
            path = db[str(idx).encode()]
            top_img_files.append(path.decode())

    return top_img_files, top_values


if __name__ == "__main__":
    text = ["Fire"]
    n_results = 6

    device = "cpu"

    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    output_file = "C:/Users/jonas/Desktop/random_numbers.csv"


    print("Loading features")
    image_features = np.load("image_features.npy")

    print("Encoding text")
    text_features = encode_text(text, model, device).cpu().numpy()

    print("Searching")
    top_img_files, top_values = search(query_features=text_features,
                                       image_features=image_features,
                                       n_results=n_results)

    print("Plotting")
    rows = math.floor(math.sqrt(n_results))
    cols = math.ceil(n_results / rows)

    plt.figure()
    for i, img_file in enumerate(top_img_files):
        plt.subplot(rows, cols, i + 1)
        img = Image.open(img_file)
        img = ImageOps.exif_transpose(img)
        plt.imshow(img)
        plt.title(f"{i + 1}: {top_values[i]:.2f}")

    plt.show()
