import logging 
import time
from os import path, makedirs
from copy import deepcopy
from datetime import datetime


from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluate.mean_average_precision import get_mean_average_precision
from evaluate.intersection_over_union import non_maximum_suppression
from evaluate.plot_detections import plot_detections
from utils import Box, Location, evaluating, format_object_locations
from data_manip.transform import get_train_val_iters
import config
from gradient_schedule import GradientSchedule


class ObjectDetector():

    def __init__(self, num_classes, rank, restore):
        self._model = self._init_pretrained_model(num_classes)
        self._rank = rank
        assert self._rank == dist.get_rank()
        self.init_training()
        if restore:
            file_path = path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
            state = torch.load(file_path)
            self._model.load_state_dict(state["model"])
            self._optimizer.load_state_dict(state["optimizer"])

    def init_training(self):
        # Reset the step and gradient schedule
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=.001)
        self._gradient_schedule = GradientSchedule(self._model)

    def _init_pretrained_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(pretrained=True, max_size=config.IMAGE_SIZE, box_nms_thresh=.5)

        torch.manual_seed(0)  # Init the same params in all processes
        box_predictor = FastRCNNPredictor(
            in_channels=model.roi_heads.box_head.fc7.out_features,
            num_classes=num_classes)
        model.roi_heads.box_predictor = box_predictor

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model = DDP(model, find_unused_parameters=True)

        return model

        

    def train(self, dataset_cls):
        training_iter, validation_iter = get_train_val_iters(dataset_cls, config.BATCH_SIZE)
        labels_dict = dataset_cls.get_labels_dict()
        summary_writer, stats_file_path = self._init_train_loggers()
        # summary_writer.add_graph(self._model, next(training_iter)[0])
        

        metrics = {}
        while not self._gradient_schedule.is_done():
            step = self._get_current_step()
            metrics.update(self._run_train_step(training_iter, step))

            if step % config.EVAL_STEPS == 0:
                metrics.update(self._run_train_eval_step(validation_iter, labels_dict, step))
                metrics.update(self._gradient_schedule.update(step))
                self._log_metrics(summary_writer, stats_file_path, metrics, step)
                self._checkpoint_model()


    def _sync_gradients(self):
        size = float(dist.get_world_size())
        for param in self._model.parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size


    def _run_train_step(self, training_iter, step):
        time_start = time.time()
        images, targets = next(training_iter)
        time_data = time.time()

        losses = self._run_model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        time_fwd = time.time() 

        self._optimizer.zero_grad()
        loss.backward()
        self._sync_gradients()
        self._optimizer.step()
        time_back = time.time()

        # dist.barrier()
        loss = loss.item()
        logging.info("\tStep: %s \tLoss: %.2f \ttime-data: %.2f \ttime-fwd: %.2f \ttime-back: %.2f",
                        step, loss, time_data-time_start, time_fwd-time_data, time_back-time_fwd)
        return {"Loss/train": loss}

    def is_master(self):
        return self._rank == 0

    def _run_train_eval_step(self, validation_iter, labels_dict, step):
        images, targets = next(validation_iter)
        
        losses = self._run_model(images, targets)
        loss = sum(losses.values()).item()  # TODO: add weighting

        ground_truths, detections = self.get_ground_truths_and_detections(images, targets, labels_dict)
        mAP = get_mean_average_precision(ground_truths, detections)
        figure = plot_detections(images[0], ground_truths[0], detections[0]) if self.is_master() else None

        return {"Loss/val": loss, "mAP/val": mAP, "Image/detections/val": figure}


    def get_ground_truths_and_detections(self, images, targets, labels_dict):   
        with evaluating(self._model):
            predictions = self._run_model(images)

        ground_truths = [format_object_locations(target_dict, labels_dict, batch_id) \
                        for batch_id, target_dict in enumerate(targets)]
        detections = [format_object_locations(detection_dict, labels_dict, batch_id) \
                        for batch_id, detection_dict in enumerate(predictions)]
        return ground_truths, detections


    def _checkpoint_model(self):
        if not self.is_master():
            return

        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict()
        }
        file_path = path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
        torch.save(state, file_path)


    def _init_train_loggers(self):
        if not self.is_master():
            return None, None

        summary_writer = SummaryWriter(log_dir=path.join(config.LOG_DIR, "tf_boards"))
        stats_filename = datetime.now().strftime('%Y%m%d_%H%M') + ".csv"
        stats_file_path = path.join(config.LOG_DIR, "stats", stats_filename)
        if not path.isdir(path.dirname(stats_file_path)):
            makedirs(path.dirname(stats_file_path))
        return summary_writer, stats_file_path


    def _log_metrics(self, summary_writer, stats_file_path, metrics, step):
        logging.info("\tStep: %s \ttrain-loss: %.2f \teval-loss: %.2f \tmAP: %.4f",
                        step, metrics["Loss/train"], metrics["Loss/val"], metrics["mAP/val"])

        # Sync the metrics on the master
        size = float(dist.get_world_size())
        for tag in list(metrics):
            if tag.startswith("*") or tag.startswith("Image"):
                continue
            metric = metrics[tag]
            tensor = torch.tensor(metric)
            dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
            metrics[tag] = tensor.item() / size

        if not self.is_master():
            return

        metrics["*Step"] = step
        metrics["*Time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        headers = [tag for tag in sorted(metrics.keys()) if not tag.startswith("Image")]
        if not path.isfile(stats_file_path):
            with open(stats_file_path, "w") as fp:
                fp.write(";\t".join(headers) + "\n")
        
        with open(stats_file_path, "a") as fp:
            fp.write(";\t".join(str(metrics[t]) for t in headers) + "\n")

        for tag, value in metrics.items():
            if tag.startswith("Image"):
                summary_writer.add_figure(tag, value, step)
            elif tag.startswith("*"):
                continue
            else:
                summary_writer.add_scalar(tag, value, step)


    def _run_model(self, inputs, targets=None):
        """ Transform the _model function into a pure function """
        return self._model(inputs, deepcopy(targets))


    def _get_current_step(self):
        params = self._optimizer.state.values()
        if params:
            return list(params)[0]["step"]
        return 0


