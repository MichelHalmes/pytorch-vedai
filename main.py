# 0 - car  
# 1 - truck  
# 2 - pickup  
# 3 - tractor  
# 4 - camping car  
# 5 - boat  
# 6 - motorcycle  
# 7 - bus  
# 8 - van  
# 9 - other  
# 10 - small  
# 11 - large  


# Data Augmentation
# https://github.com/Paperspace/DataAugmentationForObjectDetection


# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys


from torchvision import datasets, transforms, models
import numpy as np

from vedai_dataset import VedaiDataset

import utils



def main():
    train_set = VedaiDataset(for_training=True)
    eval_set = VedaiDataset(for_training=False)

    img_tensor, boxes, lables = train_set[10]
    utils.plot_sample(img_tensor, boxes, lables)
    


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    main()







