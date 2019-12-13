

# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys
import os

import click
import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from datasets.vedai import VedaiDataset
from datasets.dota import DotaDataset
from object_detector import ObjectDetector


def train_model(rank, kwargs):
    """ Distributed function"""
    num_classes = max(
        len(DotaDataset.get_labels_dict()),
        len(VedaiDataset.get_labels_dict())
    )

    detector = ObjectDetector(num_classes, rank, **kwargs)
    detector.train(DotaDataset)
    detector.train(VedaiDataset)
 
 
def init_process(rank, size, run_fn, kwargs):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    run_fn(rank, kwargs)


@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def run_distributed(**kwargs):
    SIZE = 6
    processes = []
    for rank in range(SIZE):
        p = Process(target=init_process, args=(rank, SIZE, train_model, kwargs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    run_distributed()


