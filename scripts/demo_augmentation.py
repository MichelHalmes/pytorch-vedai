import logging
import sys
from os import path
import random
from copy import copy
import random

import matplotlib.pyplot as plt

from src.datasets.vedai import VedaiDataset
from src.data_manip.transform import get_transform_collate_fn
from src.evaluate.plot_detections import plot_detections
from src.utils import format_object_locations
from src import config


def main():
    dataset = VedaiDataset(for_training=True)
    collate_fn = get_transform_collate_fn(for_training=True)
    labels_dict = dataset.get_labels_dict()

    NB_IMAGES = len(dataset)
    random.seed()
    img_idx = random.randrange(0, NB_IMAGES)
    image, target = dataset[img_idx]

    for i in range(4):
        new_image, new_target = collate_fn([(image, copy(target))])

        ground_truths = format_object_locations(new_target[0], labels_dict, img_idx)

        detections = []
        fig = plot_detections(new_image[0], ground_truths, detections, )

        fig_path = path.join(config.LOG_DIR, f"augmentation_{i}.png")
        fig.savefig(fig_path)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    main()
