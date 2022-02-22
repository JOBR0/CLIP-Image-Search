from glob import glob

import torch
import clip
from PIL import Image
from torch.utils import data

from util import get_image_extentions, ImgDataset
import pandas as pd

def encode():


    BATCH_SIZE = 8
    NUM_WORKERS = 1

    exts = get_image_extentions()
    print(exts)

    folder = "/media/jonas/DATA_SSD/OpticalFlowDatasets/MPI-Sintel-complete/test/"
    folder = "C:/Users/Jonas/Desktop/Jonas"

    folder = "//nas_enbaer/Jonas/Fotos"

    output_file = "C:/Users/Jonas/Desktop/reps.csv"

    files = []

    for ext in exts:
        files += glob(f"{folder}/**/*{ext}", recursive=True)

    print(files)

    available = clip.available_models()
    print(available)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
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
            image_features = model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features *= model.logit_scale.exp()

            features += image_features.cpu().numpy().tolist()


    df = pd.DataFrame({"path": files, "features": features})
    df.to_csv(output_file)


if __name__ == "__main__":
    encode()