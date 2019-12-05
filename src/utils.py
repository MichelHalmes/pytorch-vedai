from collections import namedtuple
from contextlib import contextmanager

Box = namedtuple("Box", ["x_min", "y_min", "x_max", "y_max"])
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