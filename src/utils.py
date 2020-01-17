from collections import namedtuple
from contextlib import contextmanager

import torch

Box = namedtuple("Box", ["x_min", "y_min", "x_max", "y_max"])
# Targets = namedtuple("Target", ["labels", "boxes"])
Location = namedtuple("Location", ["image_id", "label", "score", "box"])


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def format_object_locations(locations_dict, labels_dict, img_id="none"):
    """ Formats predictions and ground truths for metric evaluations:
        locations_dict: {boxes: tensor(x_min, y_min, x_max, y_max), lables: tensor(label_ids), <scores: tensor(score)>}
        returns: [(image_id, label_name, score, ((x_min, y_min, x_max, y_max)))]
    """
    locations = []
    if "scores" not in locations_dict:
        locations_dict["scores"] = torch.ones(locations_dict["labels"].size(), dtype=torch.float64)
    for box, label, score in zip(locations_dict["boxes"], locations_dict["labels"], locations_dict["scores"]):
        locations.append(Location(
            img_id,
            labels_dict[label.item()],
            score.item(),
            Box(*box.tolist())))
    return locations
