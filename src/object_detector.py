import logging 
import time
from os import path
from copy import deepcopy


import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from vedai_dataset import VedaiDataset
from evaluate.mean_average_precision import get_mean_average_precision
from evaluate.intersection_over_union import non_maximum_suppression
from evaluate.plot_detections import plot_detections
from utils import Box, Location, evaluating, format_object_locations
from transform.transform import get_train_val_iters
import config


class ObjectDetector():

    def __init__(self, num_classes, restore):
        self._model = self._init_pretrained_model(num_classes)
        print(self._model)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=.0001)
        if restore:
            file_path = path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
            state = torch.load(file_path)
            self._model.load_state_dict(state["model"])
            self._optimizer.load_state_dict(state["optimizer"])
        nb_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logging.info("Detector has %s trainable parameters", nb_params)


    def _init_pretrained_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(pretrained=True, max_size=config.IMAGE_SIZE, box_nms_thresh=.3)
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)

        box_predictor = FastRCNNPredictor(
            in_channels=model.roi_heads.box_head.fc7.out_features,
            num_classes=num_classes)

        model.roi_heads.box_predictor = box_predictor

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model


    def _run_model(self, inputs, targets):
        """ Transform the _model function into a pure function """
        return self._model(inputs, deepcopy(targets))
        

    def train(self, dataset_cls):
        training_iter, validation_iter = get_train_val_iters(dataset_cls, config.BATCH_SIZE)
        labels_dict = dataset_cls.get_labels_dict()
        summary_writer = SummaryWriter(log_dir=config.LOG_DIR)
        # summary_writer.add_graph(self._model, next(training_iter)[0])

        metrics = {}
        while True:
            step = self._get_current_step()
            metrics.update(self._run_train_step(training_iter, step))

            if step % config.EVAL_STEPS == 0:
                metrics.update(self._run_train_eval_step(validation_iter, labels_dict, step))
                # raise 
                self._log_metrics(summary_writer, metrics, step)
                self._checkpoint_model()


    def _run_train_step(self, training_iter, step):
        time_start = time.time()
        images, targets = next(training_iter)
        time_data = time.time()
        losses = self._run_model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        time_fwd = time.time() 
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        time_back = time.time()

        loss = loss.item()
        logging.info("\tStep: %s \tLoss: %.2f \ttime-data: %.2f \ttime-fwd: %.2f \ttime-back: %.2f",
                        step, loss, time_data-time_start, time_fwd-time_data, time_back-time_fwd)
        return {"Loss/train": loss}

    def _run_train_eval_step(self, validation_iter, labels_dict, step):
        images, targets = next(validation_iter)
        
        losses = self._run_model(images, targets)
        print(losses)
        loss = sum(losses.values())  # TODO: add weighting
        ground_truths, detections = self.get_ground_truths_and_detections(images, targets, labels_dict)

        mAP = get_mean_average_precision(ground_truths, detections)
        figure = plot_detections(images[0], ground_truths[0], detections[0])

        return {"Loss/val": loss, "mAP/val": mAP, "Image/detections/val": figure}

    def get_ground_truths_and_detections(self, images, targets, labels_dict):   
        with evaluating(self._model):
            predictions = self._run_model(images, targets) # TODO: why not targets None?

        ground_truths = [format_object_locations(target_dict, labels_dict, batch_id) \
                        for batch_id, target_dict in enumerate(targets)]
        detections = [format_object_locations(detection_dict, labels_dict, batch_id) \
                        for batch_id, detection_dict in enumerate(predictions)]
        return ground_truths, detections


    def _checkpoint_model(self):
        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict()
        }
        file_path = path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
        torch.save(state, file_path)

    @staticmethod
    def _log_metrics(writer, metrics, step):
        logging.info("\tStep: %s \ttrain-loss: %.2f \teval-loss: %.2f \tmAP: %.4f",
                        step, metrics["Loss/train"], metrics["Loss/val"], metrics["mAP/val"])

        for tag, value in metrics.items():
            if tag.startswith("Image"):
                writer.add_figure(tag, value, step)
            else:
                writer.add_scalar(tag, value, step)

    def _get_current_step(self):
        params = self._optimizer.state.values()
        if params:
            return list(params)[0]["step"]
        return 0




