import mimetypes

from PIL import Image
from torch.utils import data


def get_image_extentions():
    exts = []

    for ext, type in mimetypes.types_map.items():
        if type.split("/")[0] == "image":
            exts.append(ext)

    return exts


class ImgDataset(data.Dataset):
    def __init__(self, file_list, preprocess):
        super().__init__()

        self.image_list = file_list
        self.preprocess = preprocess

    def __getitem__(self, index):
        path = self.image_list[index]
        img = Image.open(path)

        img = self.preprocess(img)

        return img, path

    def __len__(self):
        return len(self.image_list)
