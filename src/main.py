



# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys


from torch.utils.data import DataLoader
import numpy as np

from vedai_dataset import VedaiDataset
import utils
from object_detector import ObjectDetector



def main():
    training_dataset = VedaiDataset(for_training=True)
    validation_dataset = VedaiDataset(for_training=False)

    training_loader = DataLoader(training_dataset, batch_size=20, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=20, shuffle=False)

    # img_tensor, boxes, lables = training_dataset[10]
    # utils.plot_sample(img_tensor, boxes, lables)
    labels_dict = VedaiDataset.get_labels_dict()
    ObjectDetector(num_classes=len(labels_dict))


    


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    main()






