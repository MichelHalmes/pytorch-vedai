from os import path, listdir, rename
import logging
import random

from PIL import Image

import config
from utils import Box


class MyDataset(object):
    _NAME = None
    _LABELS_DICT = None


    def __init__(self, for_training):
        self._image_ids = self._load_image_ids(for_training)


    def _load_target(self, image_id, img_size=None):
        raise NotImplementedError()


    def _annotation_path(self, image_id):
        return path.join(config.DATA_PATH.format(name=self._NAME), config.ANNOTATIONS_PATH.format(id_=image_id))


    def _load_and_filter_image_ids(self):
        images_dir = path.dirname(path.join(config.DATA_PATH.format(name=self._NAME), config.IMAGES_PATH))
        image_ids = []
        for filename in listdir(images_dir):
            if not filename.endswith(".jpg"):
                continue
            image_id = filename.split(".", 2)[0]
            annotation_path = self._annotation_path(image_id)
            if not path.exists(annotation_path) or not self._load_target(image_id, (10, 10))["labels"]: # Filter out images with only None labels or no boxes
                file_path = path.join(images_dir, filename)
                rename(file_path, file_path+".out")
            else:
                image_ids.append(image_id)
        return image_ids


    def _load_image_ids(self, for_training):
        image_ids = self._load_and_filter_image_ids()

        random.seed(0)
        random.shuffle(sorted(image_ids))
        split_idx = int(config.EVALSET_PCT*len(image_ids))
        if for_training:
            image_ids = image_ids[split_idx+1:]
        else:
            image_ids = image_ids[:split_idx]

        logging.info("Found %s images in %s-dataset", 
            len(image_ids), "train" if for_training else "eval")
        
        return image_ids


    def __getitem__(self, i):
        image_id = self._image_ids[i]
        image_path = path.join(config.DATA_PATH.format(name=self._NAME), config.IMAGES_PATH.format(id_=image_id))
        image = Image.open(image_path)
        assert image.mode == "RGB", f"Bad image {image_id} with mode {image.mode}"

        target = self._load_target(image_id, image.size)
        try:
            image, target = self._crop_to_size(image, target)
        except RecursionError:
            logging.error(image_id)
            raise

        return image, target


    def __len__(self):
        return len(self._image_ids)


    @classmethod
    def _crop_to_size(cls, image, target, recurs_cnt=0):
        # Dota contains images of differnt size, but each pixel represents the same scale
        # Make sure all images are at the same scale by croping to a maximum size
        H, W = image.size[::-1]  # PIL returns W, H
        nH, nW = config.CROP_TO_SIZE
        nH, nW = min(H, nH), min(W, nW)
        offset_y, offset_x = random.randint(0, H-nH), random.randint(0, W-nW)
        new_image = image.crop((offset_x, offset_y, offset_x+nW, offset_y+nH))
        new_boxes = [Box(b.x_min-offset_x, b.y_min-offset_y, b.x_max-offset_x, b.y_max-offset_y) for b in target["boxes"]]

        # Remove boxes outside the image
        new_target = {"boxes": [], "labels": []}
        for box, label in zip(new_boxes, target["labels"]):
            if box.x_min>=0 and box.y_min>=0 and box.x_max<=nW and box.y_max<=nH:
                new_target["boxes"].append(box)
                new_target["labels"].append(label)

        if not new_target["labels"]:
            # We removed all boxes, retry
            return cls._crop_to_size(image, target, recurs_cnt+1)
        else:
            return new_image, new_target


    @classmethod
    def get_labels_dict(cls):
        return cls._LABELS_DICT


