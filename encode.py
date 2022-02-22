from glob import glob

import torch
import clip
from PIL import Image
from torch.utils import data

from util import get_image_extentions, ImgDataset
import pandas as pd

BATCH_SIZE = 2
NUM_WORKERS = 4

exts = get_image_extentions()
print(exts)

folder = "/media/jonas/DATA_SSD/OpticalFlowDatasets/MPI-Sintel-complete/test/"

output_file = "/home/jonas/Documents/CLIP-Image-Search/reps.csv"

files = []

for ext in exts:
    files += glob(f"{folder}/**/*{ext}", recursive=True)

print(files)

available = clip.available_models()
print(available)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = ImgDataset(files, preprocess)
data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

features = []
paths = []

with torch.inference_mode():
    for i_batch, batch in enumerate(data_loader):
        print(f"{i_batch}/{len(data_loader)}")
        imgs, paths = batch
        imgs = imgs.to(device)
        feat = model.encode_image(imgs)

        features += feat.cpu().numpy().tolist()


df = pd.DataFrame({"path": files, "features": features})
df.to_csv(output_file)

# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
# with torch.inference_mode():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]