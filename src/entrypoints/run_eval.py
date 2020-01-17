import logging
import sys
from os import environ, path

from torch.utils.data import DataLoader
import torch.distributed as dist

sys.path.append(path.abspath(path.join(__file__, "../../")))

from datasets.vedai import VedaiDataset
from data_manip.transform import get_transform_collate_fn
from object_detector import ObjectDetector
from evaluate.mean_average_precision import get_mean_average_precision


def main():
    dataset = VedaiDataset(for_training=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=get_transform_collate_fn(for_training=False))
    num_classes = len(VedaiDataset.get_labels_dict())
    detector = ObjectDetector(num_classes, restore=True)
    labels_dict = dataset.get_labels_dict()

    all_ground_truths, all_detections = [], []
    mAPs = []
    for images, targets in loader:
        ground_truths, detections = detector.get_ground_truths_and_detections(images, targets, labels_dict)
        mAP = get_mean_average_precision(ground_truths, detections)
        mAPs.append(mAP)
        all_ground_truths.extend(ground_truths)
        all_detections.extend(detections)

    mAP = get_mean_average_precision(all_ground_truths, all_detections)
    logging.info("mAP for validation set: %.2f%%", mAP*100.)
    mAP = sum(mAPs) / len(mAPs)
    logging.info("Individual mAP for validation set: %.2f%%", mAP*100.)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s")

    environ["MASTER_ADDR"] = "127.0.0.1"
    environ["MASTER_PORT"] = "29503"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    main()
