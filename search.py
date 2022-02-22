import pandas as pd
from ast import literal_eval

import clip
import torch

import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def encode_text(text, model, device):
    text = clip.tokenize(text).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def load_features(device):
    output_file = "C:/Users/Jonas/Desktop/reps.csv"
    df = pd.read_csv(output_file)
    image_features = [literal_eval(f) for f in df["features"]]
    image_features = torch.tensor(image_features).to(device)
    paths = df["path"].to_numpy()
    return image_features, paths


def search(image_features, query_features, paths, n_results=6):
    with torch.inference_mode():
        print("search")

        # cosine similarity as logits
        logits_per_image = image_features @ query_features.t()

        top_values, top_indices = torch.topk(logits_per_image.squeeze(), n_results)
        top_indices = top_indices.cpu().numpy()
        top_values = top_values.cpu().numpy()

        # logits_per_text = logits_per_image.t()

    top_img_files = paths[top_indices].tolist()

    return top_img_files, top_values


if __name__ == "__main__":
    text = ["Blurry Image"]
    n_results = 6

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_features, paths = load_features(device)

    text_features = encode_text(text, model, device)

    top_img_files, top_values = search(query_features=text_features,
                                       image_features=image_features,
                                       paths=paths,
                                       n_results=n_results)

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
