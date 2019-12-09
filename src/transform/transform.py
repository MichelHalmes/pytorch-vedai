import random
from copy import copy

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
        
        new_H, new_W, _ = img.shape
        y_pad = H - new_H
        x_pad = W - new_W
        # When we scale, with a factor >1 some part of the image must be dropped,
        # We handle this via a ._pad<0 and we make sure we drop the same amount on each side of the image
        # ie the center of the image remains fixes
        # Same for factors <1, describing padding of size ._pad>0 to be added
        y_pad_1, y_pad_2 = ceil(y_pad/2.), floor(y_pad/2.)
        x_pad_1, x_pad_2 = ceil(x_pad/2.), floor(x_pad/2.)

        canvas = np.zeros((H, W, C), dtype=np.uint8)
   
        canvas[max(y_pad_1,0):min(H-y_pad_2,H), max(x_pad_1,0):min(W-x_pad_2,W), :] = \
            img[max(-y_pad_1,0):H-y_pad_1, max(-x_pad_1,0):W-x_pad_1, :]
        img = canvas
    
        targets["boxes"] *= [scale_x, scale_y, scale_x, scale_y]
        targets["boxes"] += [x_pad/2, y_pad/2, x_pad/2, y_pad/2]
        targets = clip_boxes(targets, Box(0,0,W,H))
    
        return img, targets


class RandomHSV(object):
    def __init__(self, brightness=(-10, 10), contrast=(.8, 1.5), saturation=(-10, 10), hue=(-5, 5)):
        self._brightness = brightness
        self._contrast = contrast
        self._saturation = saturation
        self._hue = hue

    def __call__(self, img, targets):
        alpha = random.uniform(*self._contrast)  # (0, 1, 3) (min, neutral, max)
        beta = random.uniform(*self._brightness)   # (-100, 0, 100)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
        img[:,:,0] += random.randint(*self._hue)
        img[:,:,1] += random.randint(*self._saturation)
        img = np.clip(img, 0, 255)
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2RGB)

        return img, targets


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
            new_img, new_targets = t(copy(img), copy(targets))
            if len(new_targets["labels"]):
                # If the transformation removes all objects from the image, we keep the original
                img, targets = new_img, new_targets

        return img, targets


def get_transform_fn(for_training):
    transforms = [
        ToNumpyArray()
    ]

    if for_training or True: # TODO
        transforms.extend([
            RandomScale(scale=.5),
            RandomHSV(),
        ])

    transforms.extend([
        ToPytorchTensor(),
    ])

    transform = Compose(transforms)

    return transform
    

