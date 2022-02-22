import pandas as pd
from ast import literal_eval

import clip
import torch

import math

import matplotlib.pyplot as plt

from PIL import Image
n_results = 6

output_file = "/home/jonas/Documents/CLIP-Image-Search/reps.csv"
df = pd.read_csv(output_file)

features = [literal_eval(f) for f in df["features"]]



text = ["a dog"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(text).to(device)
image_features = torch.tensor(features).to(device)

with torch.inference_mode():
    text_features = model.encode_text(text).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = image_features @ text_features.t()

    top_values, top_indices = torch.topk(logits_per_image.squeeze(), n_results)
    top_indices = top_indices.cpu().numpy()
    top_values = top_values.cpu().numpy()

    #logits_per_text = logits_per_image.t()


top_img_files = df["path"][top_indices].tolist()

rows = math.floor(math.sqrt(n_results))
cols = math.ceil(n_results / rows)

plt.figure()
for i, img_file in enumerate(top_img_files):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(Image.open(img_file))
    plt.title(f"{i + 1}: {top_values[i]:.2f}")

plt.show()


pass