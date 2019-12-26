from os import path
import csv

from datasets.dataset import MyDataset

import config
from utils import Box


class VedaiDataset(MyDataset):
    """ Dataset provided by: https://github.com/nikitalpopov/vedai """
    _NAME = "vedai"
    _LABELS_DICT = {
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

    def _load_target(self, image_id, img_size=None):
        annotation_path = self._annotation_path(image_id)
        with open(annotation_path) as fp:
            reader = csv.DictReader(fp, fieldnames=("label", "cx", "cy", "width", "height"), delimiter=" ")

            boxes = []
            labels = []
            W, H = img_size
            for r in reader:
                cx, cy, w, h = float(r["cx"]), float(r["cy"]), float(r["width"]), float(r["height"])
                    
                x_min = max(cx*W - w*W/2, 0)
                y_min = max(cy*H - h*H/2, 0)
                x_max = min(cx*W + w*W/2, W)
                y_max = min(cy*H + h*H/2, H)
                boxes.append(Box(x_min, y_min, x_max, y_max))
                labels.append(int(r["label"]))

        return dict(boxes=boxes, labels=labels)


if __name__ == "__main__":
    import sys
    import logging
    import numpy as np

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    training_dataset = VedaiDataset(for_training=False)

    images = []
    for i, (img, _) in enumerate(training_dataset):
        if i > 200:
            break
        images.append(img)

    images = np.stack(images)  # (N, C, H, W)

    mean = images.mean(axis=(0,2,3))
    logging.info("Pixel mean: %s", images.mean(axis=(0,2,3)))
    logging.info("Pixel std: %s", images.std(axis=(0,2,3)))
    


        




        



        


