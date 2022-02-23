from glob import glob

import torch
import clip
from PIL import Image
from torch.utils import data

from util import get_image_extentions, ImgDataset
import pandas as pd

import argparse

import time


def encode(folder, output_file, model, batch_size=2, num_workers=2, overwrite=False):
    exts = get_image_extentions()

    files = []

    for ext in exts:
        files += glob(f"{folder}/**/*{ext}", recursive=True)

    print(f"Found {len(files)} files")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(device)
    model, preprocess = clip.load(model, device=device)

    dataset = ImgDataset(files, preprocess)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  drop_last=False)

    features = []


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

    # TODO handle overwrite

    df = pd.DataFrame({"path": files, "features": features})
    df.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="//nas_enbaer/Jonas/Fotos")
    parser.add_argument("--output", type=str, default="C:/Users/Jonas/Desktop/reps.csv")
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
