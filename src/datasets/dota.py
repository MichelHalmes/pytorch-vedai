from os import path
import csv


from datasets.dataset import MyDataset

import config
from utils import Box


class DotaDataset(MyDataset):
    """ Dataset provided by: https://captain-whu.github.io/DOTA/dataset.html """
    _NAME = "dota"
    _LABELS_DICT = {
        0: "small-vehicle",
        1: "large-vehicle",
        2: None,
        3: "storage-tank",
        4: "swimming-pool",
        5: "ship",
        6: None,
        7: "tennis-court",
        8: "roundabout",
        9: "helicopter",
        10: None,
        11: "plane",
    }

    _REVERSE_LABELS_DICT = {name: id_ for id_, name in _LABELS_DICT.items() if id_ is not None}

    def _load_target(self, image_id, img_size=None):
        annotation_path = path.join(config.DATA_PATH.format(name=self._NAME), config.ANNOTATIONS_PATH.format(id_=image_id))
        with open(annotation_path) as fp:
            next(fp), next(fp)  # Skip the first two rows
            reader = csv.DictReader(fp, fieldnames=("x1","y1","x2","y2","x3","y3","x4","y4","category","difficult"), delimiter=" ")

            boxes = []
            labels = []
            for r in reader:
                label_id = self._REVERSE_LABELS_DICT.get(r["category"])
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


if __name__ == "__main__":
    import sys
    import logging
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
    

