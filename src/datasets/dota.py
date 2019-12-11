from os import path, listdir
import logging
import random
import csv
import sys

from torch.utils.data import Dataset
from PIL import Image

from utils import Box
# from transform.bbox_utils import clip_boxes


DATA_PATH = "./data/dota"
IMAGES_PATH = "images/{id_}.png"
ANNOTATIONS_PATH = "annotations/{id_}.txt"
EVALSET_PCT = .1
CROP_TO_SIZE = (1024, 1020)  # H, W

LABELS_DICT = {
    0: "small-vehicle",
    1: "large-vehicle",
    2: None,
    3: None,
    4: None,
    5: "ship",
    5: None,
    6: None,
    7: None,
    8: None,
    9: "helicopter",
    10: None,
    11: "plane",
}

REVERSE_LABELS_DICT = {name: id_ for id_, name in LABELS_DICT.items() if id_ is not None}


class DotaDataset(Dataset):
    """ Dataset provided by: https://captain-whu.github.io/DOTA/dataset.html """

    def __init__(self, for_training):
        self._image_ids = self._load_image_ids(for_training)

    def _load_image_ids(self, for_training):
        images_dir = path.dirname(path.join(DATA_PATH, IMAGES_PATH))
        image_ids = [filename.split(".", 2)[0] for filename in sorted(listdir(images_dir)) if filename.endswith(".png")]
        image_ids = [id_ for id_ in image_ids if self._load_target(id_)["labels"]]  # Filter out images with only None labels

        random.seed(0)
        random.shuffle(image_ids)
        split_idx = int(EVALSET_PCT*len(image_ids))
        if for_training:
            image_ids = image_ids[split_idx+1:]
        else:
            image_ids = image_ids[:split_idx]

        logging.info("Found %s images in %s-dataset", 
            len(image_ids), "train" if for_training else "eval")
        
        return image_ids


    def __getitem__(self, i):
        image_id = self._image_ids[i]
        image_path = path.join(DATA_PATH, IMAGES_PATH.format(id_=image_id))
        image = Image.open(image_path)
        # image = image.convert("RGB")

        target = self._load_target(image_id)

        H, W = image.size[::-1]  # PIL returns W, H
        nH, nW = CROP_TO_SIZE
        nH, nW = min(H, nH), min(W, nW)
        offset_y, offset_x = random.randint(0, H-nH), random.randint(0, W-nW)


        image = image.crop((offset_x, offset_y, offset_x+nW, offset_y+nH))
        target["boxes"] = [Box(b.x_min-offset_x, b.y_min-offset_y, b.x_max-offset_x, b.y_max-offset_y) for b in target["boxes"]]
        # The boxes that have moved out of the image will be removed by the transform process

        return image, target


    def __len__(self):
        return len(self._image_ids)


    def _load_target(self, image_id):
        annotation_path = path.join(DATA_PATH, ANNOTATIONS_PATH.format(id_=image_id))
        with open(annotation_path) as fp:
            next(fp), next(fp)
            reader = csv.DictReader(fp, fieldnames=("x1","y1","x2","y2","x3","y3","x4","y4","category","difficult"), delimiter=" ")

            boxes = []
            labels = []
            for r in reader:
                label_id = REVERSE_LABELS_DICT.get(r["category"])
                if label_id is None:
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = [float(r[k]) for k in ("x1","y1","x2","y2","x3","y3","x4","y4")]
                    
                x_min = min(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                x_max = max(x1, x2, x3, x4)
                y_max = max(y1, y2, y3, y4)
                boxes.append(Box(x_min, y_min, x_max, y_max))
                labels.append(label_id)

        return dict(boxes=boxes, labels=labels)


    @staticmethod
    def get_labels_dict():
        return LABELS_DICT


if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    training_dataset = DotaDataset(for_training=False)

    images = []
    for i, (img, _) in enumerate(training_dataset):
        if i > 2:
            break
        images.append(img)

    images = np.stack(images)  # (N, C, H, W)

    mean = images.mean(axis=(0,2,3))
    logging.info("Pixel mean: %s", images.mean(axis=(0,2,3)))
    logging.info("Pixel std: %s", images.std(axis=(0,2,3)))
    

