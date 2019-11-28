import logging 
import time
from itertools import cycle
from os import path
from contextlib import contextmanager


import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

from vedai_dataset import VedaiDataset
from metrics.mean_average_precision import get_mean_average_precision



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


EVAL_STEPS = 10
CHECKPOINT_DIR = "./data/model"
CHECKPOINT_NAME = "model.pth.tar"

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
        model = fasterrcnn_resnet50_fpn(pretrained=True, max_size=100)  # TODO:
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

    def train(self, training_loader, validation_loader):
        training_iter = cycle(iter(training_loader))
        validation_iter = cycle(iter(validation_loader))

        step = 0
        while True:
            step += 1
            train_loss = self._run_train_step(training_iter, step)

            if step % EVAL_STEPS == 1 or True: # TODO
                val_loss, mAP = self._run_train_eval_step(validation_iter, step)
                self._checkpoint_model()
                logging.info("\tStep: %s \ttrain-loss: %.2f \teval-loss: %.2f \tmAP: %.2f",
                        step, train_loss, val_loss, mAP)

    def _run_train_step(self, training_iter, step):
        images, targets = next(training_iter)
        time_start = time.time()
        losses = self._model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        time_loss = time.time() 
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        loss = loss.item()
        logging.info("\tStep: %s \tLoss: %.2f \ttime-loss: %.2f \ttime-optimize: %.2f",
                        step, loss, time_loss-time_start, time.time()-time_loss)
        return loss

    def _run_train_eval_step(self, validation_iter, step):
        images, targets = next(validation_iter)

        losses = self._model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        with evaluating(self._model):
            detections = self._model(images, targets)

        labels_dict = VedaiDataset.get_labels_dict()
        ground_truths = self._format_object_locations(targets, labels_dict)
        detections = self._format_object_locations(detections, labels_dict)
        mAP = get_mean_average_precision(ground_truths, detections)
        return loss, mAP


    @staticmethod
    def _format_object_locations(locations_dict, labels_dict):
        """ Formats preidctions and ground truths for metric evaluations:
            locations_dict: [{boxes: tensor(x_min, y_min, x_max, y_max), lables: tensor(label_ids), <scores: tensor(score)>}]
            returns: [(image_id, label_name, score, ((x_min, y_min, x_max, y_max)))]
        """
        locations = []
        for b, locations_dict in enumerate(locations_dict):
            if not "scores" in locations_dict:
                locations_dict["scores"] = torch.ones(locations_dict["labels"].size(), dtype=torch.float64)
            for box, label, score in zip(locations_dict["boxes"], locations_dict["labels"], locations_dict["scores"]):
                locations.append((
                    f"batch_idx_{b}",
                    labels_dict[label.item()],
                    score.item(),
                    tuple(box.tolist())))

        return locations

    def _checkpoint_model(self):
        import copy
        state = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict()
        }
        file_path = path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        torch.save(state, file_path)



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

        



