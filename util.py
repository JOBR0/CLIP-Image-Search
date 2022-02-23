import mimetypes

from PIL import Image, ImageOps
from torch.utils import data


def get_image_extentions():
    exts = []

    for ext, type in mimetypes.types_map.items():
        if type.split("/")[0] == "image":
            exts.append(ext)

    return exts


def filtered_collate(batch):
    """Filters out None values from a batch if images failed to load"""
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


class ImgDataset(data.Dataset):
    def __init__(self, file_list, preprocess):
        super().__init__()

        self.image_list = file_list
        self.preprocess = preprocess

    def __getitem__(self, index):
        path = self.image_list[index]
        try:
            img = Image.open(path)
        except:
            print(f"Error opening image: {path}")
            with open("failed_images.txt", "a") as f:
                f.write(path + "\n")
            # return None which will be filtered out in filtered_collate
            return None

        # Apply stored rotation
        img = ImageOps.exif_transpose(img)

        img = self.preprocess(img)

        return img, path

    def __len__(self):
        return len(self.image_list)
