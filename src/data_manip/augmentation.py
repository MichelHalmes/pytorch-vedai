import random
import math

import numpy as np
import cv2

from .bbox_utils import clip_boxes
from ..utils import Box


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


SYMMETRIES = [
    (1, 1),  # No flip
    (-1, 1),  # Horizontal axis
    (1, -1),  # Vertical axis
    (-1, -1),  # Diagonal axis
]


class RandomAxisFlip(object):
    def __init__(self):
        pass

    def __call__(self, img, targets):
        step_y, step_x = random.choice(SYMMETRIES)
        img = img[::step_y, ::step_x, :]

        boxes = targets["boxes"]
        H, W, _ = img.shape
        if step_y == -1:
            boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
        if step_x == -1:
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        targets["boxes"] = boxes

        return img, targets


class RandomRotate(object):

    def __init__(self, angle=20):
        self._angle = angle

    def __call__(self, img, targets):
        # PART 1: GET ROTATION MATRIX
        angle = random.uniform(-self._angle, self._angle)
        H, W = img.shape[:2]
        cX, cY = W//2, H//2

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((H*sin) + (W*cos))
        nH = int((H*cos) + (W*sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW/2) - cX
        M[1, 2] += (nH/2) - cY

        # PART 2: ROTATE IMAGE
        # perform the actual rotation and return the image
        img = cv2.warpAffine(img, M, (nW, nH))

        # PART 3: ROTATE BOXES
        boxes = targets["boxes"]

        # Compute all 4 corners of the box
        width, height = boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]
        x1, y1 = boxes[:,0], boxes[:,1]
        corners = np.vstack((x1,y1, x1+width,y1, x1,y1+height, x1+width,y1+height)).T  # x1,y1,x2,y2,x3,y3,x4,y4
        corners = corners.reshape(-1,2)

        # Apply rotation to all corners
        corners = np.hstack((corners, np.ones((corners.shape[0],1))))
        corners = np.dot(M,corners.T).T
        corners = corners.reshape(-1,8)

        # Compute new bounding boxes
        x_, y_ = corners[:,[0,2,4,6]], corners[:,[1,3,5,7]]
        boxes = np.vstack((np.min(x_,1), np.min(y_,1), np.max(x_,1), np.max(y_,1))).T  # (xmin, ymin, xmax, ymax)

        # PART 4: ADJUST FOR SCALE CHANGE
        # Resize the image and boxes to original dimensions
        img = cv2.resize(img, (W,H))
        scale_x, scale_y = nW/W, nH/H
        boxes /= [scale_x, scale_y, scale_x, scale_y]
        targets["boxes"] = boxes

        return img, targets


class RandomShear(object):
    """ shear_factor: float, the image is sheared horizontally by a factor drawn
        randomly from a range (0, `shear_factor`).
    """
    def __init__(self, shear_factor=0.2):
        self._shear_factor = shear_factor

    def __call__(self, img, targets):
        H, W, _ = img.shape
        shear_factor = random.random() * self._shear_factor
        M = np.array([[1, shear_factor, 0],[0,1,0]])
        nW = int(W + shear_factor*H)

        img = cv2.warpAffine(img, M, (nW, H))
        img = cv2.resize(img, (W, H))

        boxes = targets["boxes"]
        boxes[:,[0,2]] += boxes[:,[1,3]] * shear_factor
        scale_x = nW / W
        boxes /= [scale_x, 1, scale_x, 1]
        targets["boxes"]

        return img, targets


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
        canvas = np.zeros((H, W, C))

        canvas[max(y_pad_1,0):min(H-y_pad_2,H), max(x_pad_1,0):min(W-x_pad_2,W), :] = \
            img[max(-y_pad_1,0):H-y_pad_1, max(-x_pad_1,0):W-x_pad_1, :]
        img = canvas

        targets["boxes"] *= [scale_x, scale_y, scale_x, scale_y]
        targets["boxes"] += [x_pad/2, y_pad/2, x_pad/2, y_pad/2]
        targets = clip_boxes(targets, Box(0,0,W,H))

        return img, targets


class RandomTranslate(object):

    def __init__(self, rel_step=.1):
        self._rel_step = rel_step

    def __call__(self, img, targets):
        H, W, C = img.shape
        move_y = int(random.uniform(-self._rel_step, self._rel_step) * H)
        move_x = int(random.uniform(-self._rel_step, self._rel_step) * W)

        canvas = np.zeros((H, W, C), dtype=np.float)

        canvas[max(move_y,0):min(H+move_y, H), max(move_x,0):min(W+move_x, W), :] = \
            img[max(-move_y,0):min(H-move_y, H), max(-move_x,0):min(W-move_x, W), :]
        img = canvas

        targets["boxes"] += [move_x, move_y, move_x, move_y]
        targets = clip_boxes(targets, Box(0,0,W,H))

        return img, targets


class Normalize(object):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L228
    _IMAGE_MEAN = [0.485*255, 0.456*255, 0.406*255]
    _IMAGE_STD = [0.229*255, 0.224*255, 0.225*255]

    def __init__(self, forward=True):
        """ Some transformations dd a black background
            This makes sure it end up as grey ie the pixel average
        """
        self._forward = forward

    def __call__(self, img, targets):
        if self._forward:
            img = img.astype("float32")
            img -= self._IMAGE_MEAN
            img /= self._IMAGE_STD

        else:
            img *= self._IMAGE_STD
            img += self._IMAGE_MEAN
            img = img.astype("uint8")

        return img, targets
