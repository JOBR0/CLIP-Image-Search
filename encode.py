import os

from glob import glob

import torch
import clip
from PIL import Image
from torch.utils import data

from util import get_image_extentions, ImgDataset, filtered_collate
import pandas as pd

import argparse

import time


def encode(folder, output_file, model, batch_size=2, num_workers=2, overwrite=False):
    exts = get_image_extentions()

    # Also consider for example .JPG instead of .jpg
    exts_upper = [ext.upper() for ext in exts]
    exts = exts_upper + exts

    files = []

    print("Searching for images in {}".format(folder))
    for ext in exts:
        files += glob(f"{folder}/**/*{ext}", recursive=True)

    print(f"Found {len(files)} files")

    # check if output file exists
    if not os.path.exists(output_file) or overwrite:
        print("Creating output file")
        df = pd.DataFrame(columns=["path", "encoding"])
        df.to_csv(output_file, index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    model, preprocess = clip.load(model, device=device)

    dataset = ImgDataset(files, preprocess)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  drop_last=False, collate_fn=filtered_collate)

    features = []
    feat_paths = []

    batches_before_save = 250

    with torch.inference_mode():
        for i_batch, batch in enumerate(data_loader):
            if i_batch % 10 == 0:
                if i_batch > 0:
                    elapsed = time.time() - start
                    imgs_per_sec = batch_size / elapsed
                    print(f"Img/sec: {imgs_per_sec}")
                start = time.time()
                print(f"{i_batch}/{len(data_loader)}")

            imgs, paths = batch
            imgs = imgs.to(device)
            image_features = model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features *= model.logit_scale.exp()

            features += image_features.cpu().numpy().tolist()
            feat_paths += paths

            if i_batch + 1 % batches_before_save == 0 or i_batch + 1 == len(data_loader):
                print("Saving")
                df = pd.DataFrame({"path": feat_paths, "features": features})
                df.to_csv(output_file, index=False, mode="a", header=False)

                features = []
                feat_paths = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, required=True, help="Folder to search for images")
    parser.add_argument("--output", type=str, default="./encoded.csv")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    available_models = clip.available_models()
    print(available_models)

    encode(
        folder=args.folder,
        output_file=args.output,
        model=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite
    )
