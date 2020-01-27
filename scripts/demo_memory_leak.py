import os
import gc
import random

import torch
import psutil
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# https://github.com/pytorch/vision/issues/1689

def main():
    process = psutil.Process(os.getpid())
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # model.eval() # Gives no issues

    for i in range(10):
        size = random.randint(700, 900)
        images = torch.rand([1, 3, size, 800])
        # images = torch.rand([1, 3, size, size]) # Gives no issues
        targets = [{'boxes': torch.tensor([[10., 20., 30., 40.]]), 'labels': torch.tensor([1])}]
        model(images, targets)
        gc.collect()
        print("Current memory: ", process.memory_info()[0] / float(2**20))

if __name__ == "__main__":
    main()
