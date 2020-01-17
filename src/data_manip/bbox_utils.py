import numpy as np


def get_areas(boxes):
    return (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])


def clip_boxes(targets, clip, max_lost_area_pct=.25):
    """
    clip: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.
    """
    boxes = targets["boxes"]
    original_area = get_areas(boxes)

    x_min = np.minimum(np.maximum(boxes[:,0], clip.x_min), clip.x_max).reshape(-1,1)
    x_max = np.minimum(np.maximum(boxes[:,2], clip.x_min), clip.x_max).reshape(-1,1)
    y_min = np.minimum(np.maximum(boxes[:,1], clip.y_min), clip.y_max).reshape(-1,1)
    y_max = np.minimum(np.maximum(boxes[:,3], clip.y_min), clip.y_max).reshape(-1,1)
    boxes = np.hstack((x_min, y_min, x_max, y_max))

    lost_areas_pct = get_areas(boxes)/original_area -1.
    keep = lost_areas_pct > -max_lost_area_pct

    targets["boxes"] = boxes[keep]
    targets["labels"] = targets["labels"][keep]

    return targets
