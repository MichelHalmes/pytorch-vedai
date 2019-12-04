import logging 
import time
from itertools import cycle
from os import path
from copy import deepcopy
from contextlib import contextmanager


import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from vedai_dataset import VedaiDataset
from metrics.mean_average_precision import get_mean_average_precision
from metrics.plot_detections import plot_detections



@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


class FastRcnnBoxPredictor(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


EVAL_STEPS = 20
BATCH_SIZE = 4
CHECKPOINT_DIR = "./data/model"
CHECKPOINT_NAME = "model.pth.tar"
LOG_DIR = "./data/logs/tf_boards"

class ObjectDetector():

    def __init__(self, num_classes, restore):
        self._model = self._init_pretrained_model(num_classes)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.0001)
        if restore:
            file_path = path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
            state = torch.load(file_path)
            self._model.load_state_dict(state["model"])
            self._optimizer.load_state_dict(state["optimizer"])

        nb_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logging.info("Detector has %s trainable parameters", nb_params)

    def _init_pretrained_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(pretrained=True, max_size=300)  # TODO:
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)

        box_predictor = FastRcnnBoxPredictor(
            in_channels=model.roi_heads.box_head.fc7.out_features,
            num_classes=num_classes)

        model.roi_heads.box_predictor = box_predictor

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # url = 'https://images.fineartamerica.com/images-medium-large-5/dog-and-cat-driving-car-through-snowy-john-danielsjohan-de-meester.jpg'
        # response = requests.get(url, stream = True)
        # image = Image.open(response.raw)
        # transf = transforms.ToTensor()
        # img_tensor = transf(image)
        # model.eval()
        # output = model([img_tensor])
        # model.train()
        # print(output)

        return model

    def _run_model(self, inputs, targets):
        """ Transform the _model function into a pure function """
        return self._model(inputs, deepcopy(targets))
        

    def train(self, training_dataset, validation_dataset):
        training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=training_dataset.collate_fn)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=validation_dataset.collate_fn)
        training_iter = cycle(iter(training_loader))
        validation_iter = cycle(iter(validation_loader))
        labels_dict = validation_dataset.get_labels_dict()

        summary_writer = SummaryWriter(log_dir=LOG_DIR)
        # summary_writer.add_graph(self._model, next(training_iter)[0])

        step = 0
        metrics = {}
        while True:
            metrics.update(self._run_train_step(training_iter, step))

            if step % EVAL_STEPS == 0:
                metrics.update(self._run_train_eval_step(validation_iter, labels_dict, step))
                self._log_metrics(summary_writer, metrics, step)
                self._checkpoint_model()
            
            step += 1


    def _run_train_step(self, training_iter, step):
        images, targets = next(training_iter)
        time_start = time.time()
        losses = self._run_model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        time_loss = time.time() 
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        loss = loss.item()
        logging.info("\tStep: %s \tLoss: %.2f \ttime-loss: %.2f \ttime-optimize: %.2f",
                        step, loss, time_loss-time_start, time.time()-time_loss)
        return {"Loss/train": loss}

    def _run_train_eval_step(self, validation_iter, labels_dict, step):
        images, targets = next(validation_iter)
        
        losses = self._run_model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        ground_truths, detections = self.get_ground_truths_and_detections(images, targets, labels_dict)

        mAP = 1 # get_mean_average_precision(ground_truths, detections)
        figure = plot_detections(images[0], ground_truths[0], detections[0])

        return {"Loss/val": loss, "mAP/val": mAP, "Image/detections/val": figure}

    def get_ground_truths_and_detections(self, images, targets, labels_dict):   
        with evaluating(self._model):
            predictions = self._run_model(images, targets) # TODO: why not targets None?

        ground_truths = [self._format_object_locations(target_dict, labels_dict, batch_id) \
                        for batch_id, target_dict in enumerate(targets)]
        detections = [self._format_object_locations(detection_dict, labels_dict, batch_id) \
                        for batch_id, detection_dict in enumerate(predictions)]
        return ground_truths, detections

    @staticmethod
    def _format_object_locations(locations_dict, labels_dict, img_id="none"):
        """ Formats preidctions and ground truths for metric evaluations:
            locations_dict: {boxes: tensor(x_min, y_min, x_max, y_max), lables: tensor(label_ids), <scores: tensor(score)>}
            returns: [(image_id, label_name, score, ((x_min, y_min, x_max, y_max)))]
        """
        locations = []
        if not "scores" in locations_dict:
            locations_dict["scores"] = torch.ones(locations_dict["labels"].size(), dtype=torch.float64)
        for box, label, score in zip(locations_dict["boxes"], locations_dict["labels"], locations_dict["scores"]):
            locations.append((
                img_id,
                labels_dict[label.item()],
                score.item(),
                tuple(box.tolist())))

        return locations


    def _checkpoint_model(self):
        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict()
        }
        file_path = path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        torch.save(state, file_path)

    @staticmethod
    def _log_metrics(writer, metrics, step):
        logging.info("\tStep: %s \ttrain-loss: %.2f \teval-loss: %.2f \tmAP: %.2f",
                        step, metrics["Loss/train"], metrics["Loss/val"], metrics["mAP/val"])

        for tag, value in metrics.items():
            if tag.startswith("Image"):
                writer.add_figure(tag, value, step)
            else:
                writer.add_scalar(tag, value, step)




    # def _compute_losses_and_predictions(self, images, targets):
    #     """ The torchvision class FasterRCNN only returns the loss in training mode.
    #         We also want the
    #     """
    #     with torch.no_grad():
    #         original_image_sizes = [tuple(img.shape[-2:]) for img in images] 
    #         images, targets = self._model.transform(images, targets)
    #         features = self._model.backbone(images.tensors)
    #         proposals, proposal_losses = self._model.rpn(images, features, targets)
    #         detections, detector_losses = self._model.roi_heads(features, proposals, images.image_sizes, targets)
    #         detections = self._model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    #         losses = {**detector_losses, **proposal_losses} 
    #         return losses, detections

        



