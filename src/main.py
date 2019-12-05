



# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys

import click
import numpy as np

from vedai_dataset import VedaiDataset
from object_detector import ObjectDetector


@click.command()
@click.option('--restore/--no-restore', default=True, help='Reinititalize the model or restore previous checkpoint')
def train_model(restore):
    training_dataset = VedaiDataset(for_training=True)
    validation_dataset = VedaiDataset(for_training=False)

    num_classes = len(training_dataset.get_labels_dict())
    detector = ObjectDetector(num_classes, restore)

    detector.train(training_dataset, validation_dataset)
 

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    train_model()







