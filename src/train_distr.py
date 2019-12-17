

# More data
# https://en.wikipedia.org/wiki/Overhead_Imagery_Research_Data_Set

import logging
import sys
import os
import time

import click
import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from datasets.vedai import VedaiDataset
from datasets.dota import DotaDataset
from object_detector import ObjectDetector
import config


@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def train_model(restore):
    """ Distributed function"""
    num_classes = max(
        len(DotaDataset.get_labels_dict()),
        len(VedaiDataset.get_labels_dict())
    )

    detector = ObjectDetector(num_classes, restore)
    detector.train(DotaDataset)
    detector.init_training()
    detector.train(VedaiDataset)
 
 
def init_process(rank, size, run_fn):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=size)
    try:
        run_fn()
    except KeyboardInterrupt:
        logging.info("Aborted")
    finally:
        logging.info("Stopping")
        dist.destroy_process_group()


def run_distributed():
    size = config.NB_PROCESSES
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, train_model))
        p.start()
        processes.append(p)

    while all(p.is_alive() for p in processes):
        time.sleep(5)
    
    for p in processes:
        p.kill() 
        p.join()
    logging.info("Main process exit")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    run_distributed()


