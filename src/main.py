



# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys


import numpy as np

from vedai_dataset import VedaiDataset
import utils
from object_detector import ObjectDetector



def main():
    training_dataset = VedaiDataset(for_training=True)
    validation_dataset = VedaiDataset(for_training=False)

    labels_dict = VedaiDataset.get_labels_dict()
    detector = ObjectDetector(num_classes=len(labels_dict), restore=False)

    detector.train(training_dataset, validation_dataset)
 


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    main()







