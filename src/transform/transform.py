import random

import numpy as np
import cv2

import torch
from torchvision import transforms as tv_transforms

from transform.bbox_utils import clip_boxes
from utils import Box
import math

def ceil(x): return int(math.ceil(x))
def floor(x): return int(math.floor(x))


class RandomScale(object):
   
    def __init__(self, scale=0.2, symmetric=False):
        """ scale: float :the image is scaled by a factor drawn 
                randomly from a range (1 - `scale` , 1 + `scale`).
        """
        self._scale = scale
        self._symmetric = symmetric

    def __call__(self, img, targets):
        if self._symmetric:
            scale_y = 1 + random.uniform(-self._scale, self._scale)
            scale_x = scale_y
        else:
            scale_y = 1 + random.uniform(-self._scale, self._scale)
            scale_x = 1 + random.uniform(-self._scale, self._scale)

        H, W, C = img.shape
        img = cv2.resize(img, None, fx=scale_x, fy=scale_y)
        
        y_pad = H - int(min(scale_y,1)*H)
        x_pad = W - int(min(scale_x,1)*W)
        y_min, y_max = ceil(y_pad/2.), H-floor(y_pad/2.)
        x_min, x_max = ceil(x_pad/2.), W-floor(x_pad/2.)

        canvas = np.zeros((H, W, C), dtype=np.uint8)
        canvas[y_min:y_max, x_min:x_max, :] = img[:H-y_pad, :W-x_pad, :]
        img = canvas
    
        targets["boxes"] *= [scale_x, scale_y, scale_x, scale_y]
        targets["boxes"] -= [(scale_x-1)*W/2, (scale_y-1)*H/2, (scale_x-1)*W/2, (scale_y-1)*H/2]
        targets = clip_boxes(targets, Box(x_min, y_min, x_max, y_max))
    
        return img, targets


class RandomHSV(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self._transform = tv_transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, targets):
        return self._transform(img), targets


class ToNumpyArray(object):
    def __init__(self):
        pass
    
    def __call__(self, img, targets):
        img = np.asarray(img)
        targets["boxes"] = np.asarray(targets["boxes"])
        targets["labels"] = np.asarray(targets["labels"])
        return img, targets


class ToPytorchTensor(object):
    def __init__(self):
        self._img_transform = tv_transforms.ToTensor()
    
    def __call__(self, img, targets):
        targets = {
            "boxes": torch.FloatTensor(targets["boxes"]), # (n_objects, 4)
            "labels": torch.LongTensor(targets["labels"]) # (n_objects)
        }
        img = self._img_transform(img)
        return img, targets


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, targets):
        for t in self.transforms:
            img, targets = t(img, targets)
        return img, targets


def get_transform_fn(for_training):
    transforms = [
        ToNumpyArray()
    ]

    if for_training or True: # TODO
        transforms.extend([
            RandomScale(scale=.5),
            # RandomHSV(brightness=.5, contrast=.5, saturation=.5)
        ])

    transforms.extend([
        ToPytorchTensor(),
    ])

    transform = Compose(transforms)

    return transform
    

