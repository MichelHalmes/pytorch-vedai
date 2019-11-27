from os import path, listdir
import logging
import random
import csv

from torch.utils.data import Dataset
import torch
from PIL import Image


from transform import get_transform_fn



DATA_PATH = "./data/vedai"
IMAGES_PATH = "images/{id_}.jpg"
ANNOTATIONS_PATH = "annotations/{id_}.txt"
EVALSET_PCT = .2

LABELS_DICT = {
    0: "car",
    1: "truck",
    2: "pickup",
    3: "tractor",
    4: "camping car",
    5: "boat",
    6: "motorcycle",
    7: "bus",
    8: "van",
    9: "other",
    10: "small",
    11: "large",
}



class VedaiDataset(Dataset):
    """ Dataset provided by: https://github.com/nikitalpopov/vedai """

    def __init__(self, for_training):
        self._image_ids = self._load_image_ids(for_training)
        self._transform = get_transform_fn(for_training) 

    def _load_image_ids(self, for_training):
        images_dir = path.dirname(path.join(DATA_PATH, IMAGES_PATH))
        image_ids = [filename.split(".", 2)[0] for filename in listdir(images_dir) if filename.endswith(".jpg")]
        
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
        # image = image.convert("RGB") ???????

        annotation_path = path.join(DATA_PATH, ANNOTATIONS_PATH.format(id_=image_id))
        with open(annotation_path) as fp:
            reader = csv.DictReader(fp, fieldnames=("label", "cx", "cy", "width", "height"), delimiter=" ")

            boxes = []
            labels = []
            img_width, img_height, _ = image.shape
            for r in reader:
                cx, cy, w, h = float(r["cx"]), float(r["cy"]), float(r["width"]), float(r["height"])
                x_min = cx*img_width - w*img_width/2
                y_min = cy*img_height - h*img_height/2
                x_max = cx*img_width + w*img_width/2
                y_max = cy*img_height + h*img_height/2
                boxes.append(x_min, y_min, x_max, y_max)

                labels.append(int(r["lable"]))


        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)

        return self._transform(image, boxes, labels)


    def __len__(self):
        return len(self._image_ids)

    @staticmethod
    def get_labels_dict():
        return LABELS_DICT



        



        


