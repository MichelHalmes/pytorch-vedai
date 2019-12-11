



# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys

import click
import numpy as np

from datasets.vedai import VedaiDataset
from datasets.dota import DotaDataset
from object_detector import ObjectDetector


@click.command()
@click.option('--restore/--no-restore', default=True, help='Reinititalize the model or restore previous checkpoint')
def train_model(restore):
    num_classes = len(DotaDataset.get_labels_dict())
    detector = ObjectDetector(num_classes, restore)

    detector.train(DotaDataset)
 

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train_model()







