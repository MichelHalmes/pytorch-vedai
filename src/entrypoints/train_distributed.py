import logging
import sys
from os import environ, path
import time

import click
import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from ..datasets.vedai import VedaiDataset
from ..datasets.dota import DotaDataset
from ..object_detector import ObjectDetector
from .. import config


@click.command()
@click.option("--restore/--no-restore", default=True, help="Reinititalize the model or restore previous checkpoint")
def train_model(restore):
    """ Distributed function"""
    num_classes = max(
        len(DotaDataset.get_labels_dict()),
        len(VedaiDataset.get_labels_dict())
    )

    detector = ObjectDetector(num_classes, restore)
    detector.train(DotaDataset, config.INIT_SCHEDULE)
    detector.init_optimizer()
    detector.train(VedaiDataset, config.TRAINED_SCHEDULE)


def init_process(rank, size, run_fn):
    """ Initialize the distributed environment. """
    environ["MASTER_ADDR"] = "127.0.0.1"
    environ["MASTER_PORT"] = "29501"
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


def main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    run_distributed()

if __name__ == "__main__":
    main()
