from copy import copy
from itertools import cycle

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torchvision import transforms as tv_transforms

from data_manip.bbox_utils import clip_boxes
from data_manip.augmentation import *
from utils import Box



class ToNumpyArray(object):
    
    def __call__(self, img, targets):
        img = np.array(img)
        targets["boxes"] = np.asarray(targets["boxes"])
        targets["labels"] = np.asarray(targets["labels"])
        return img, targets

class ClipBoxes(object):
    """ Limits all incoming boxes to be contained in the image """

    def __call__(self, img, targets):
        H, W, _ = img.shape
        targets = clip_boxes(targets, Box(0,0,W,H))
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
        ToNumpyArray(),
        ClipBoxes(),
    ]

    if for_training or True: # TODO
        transforms.extend([
            RandomHSV(),
            Normalize(forward=True),
            RandomAxisFlip(),
            RandomRotate(),
            RandomShear(),
            RandomScale(),
            RandomTranslate(),
            Normalize(forward=False),
        ])

    transforms.extend([
        ToPytorchTensor(),
    ])

    transform = Compose(transforms)

    return transform


def _match_batch_img_sizes(batch):
    # Resize so that all images have the same size
    nH = max(img.size[1] for img, _ in batch)
    nW = max(img.size[0] for img, _ in batch)
    new_batch = []
    for image, target in batch:
        H, W = image.size[::-1]  # PIL returns W, H
        scale_y, scale_x = nH/H, nW/W
        image = image.resize((nW, nH))
        target["boxes"] = [Box(b.x_min*scale_x, b.y_min*scale_y, b.x_max*scale_x, b.y_max*scale_y) for b in target["boxes"]]
        new_batch.append((image, target))
    return new_batch


def get_transform_collate_fn(for_training):
    transform = get_transform_fn(for_training)
    def collate(batch):
        batch = _match_batch_img_sizes(batch)
        images = []
        targets = []
        for image, target in batch:
            image, target = transform(image, target)
            images.append(image)
            targets.append(target)

        images = torch.stack(images, dim=0)

        return images, targets

    return collate


def loop_forever(loader):
    while True:
        data_iter = iter(loader)
        for data in data_iter:
            yield data
        loader.sampler.epoch += 1  # Important for shuffling the DistributedSampler


def get_train_val_iters(dataset_cls, batch_size):
    def get_loader(for_training):
        dataset = dataset_cls(for_training)
        dist_sampler = DistributedSampler(dataset, shuffle=for_training)
        return DataLoader(dataset, 
                        batch_size=batch_size,  # num_workers=1 if for_training else 0
                        sampler=dist_sampler, # shuffle=for_training, 
                        collate_fn=get_transform_collate_fn(for_training))
    training_loader = get_loader(True)
    validation_loader = get_loader(False)
    training_iter = loop_forever(training_loader)
    validation_iter = loop_forever(validation_loader)
    return training_iter, validation_iter
    

