import logging 
import time
from itertools import cycle
import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import  transforms

from torch.hub import load_state_dict_from_url


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

class ObjectDetector():
    def __init__(self, num_classes):
        self._model = self._init_pretrained_model(num_classes)
        nb_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logging.info("Detector has %s trainable parameters", nb_params)

    def _init_pretrained_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(pretrained=True, max_size=200)  # TODO:
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
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.0001)

        step = 0
        while True:
            step += 1
            self._run_train_step(training_iter, optimizer, step)

            if step % EVAL_STEPS == 1 or True: # TODO
                self._run_train_eval_step(validation_iter, step)

    def _run_train_step(self, training_iter, optimizer, step):
        images, targets = next(training_iter)
        time_start = time.time()
        losses = self._model(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        time_loss = time.time() 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info("\tStep: %s \tLoss: %.2f \ttime-loss: %.2f \ttime-optimize: %.2f",
                        step, loss.item(), time_loss-time_start, time.time()-time_loss)

    def _run_train_eval_step(self, validation_iter, step):
        images, targets = next(validation_iter)

        self._model.eval()
        losses, predictions = self._compute_losses_and_predictions(images, targets)
        loss = sum(losses.values())  # TODO: add weighting
        self._model.train()
        print(predictions)


    def _compute_losses_and_predictions(self, images, targets):
        """ The torchvision class FasterRCNN only returns the loss in training mode.
            We also want the
        """
        with torch.no_grad():
            original_image_sizes = [tuple(img.shape[-2:]) for img in images] 
            images, targets = self._model.transform(images, targets)
            features = self._model.backbone(images.tensors)
            proposals, proposal_losses = self._model.rpn(images, features, targets)
            detections, detector_losses = self._model.roi_heads(features, proposals, images.image_sizes, targets)
            detections = self._model.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            losses = {**detector_losses, **proposal_losses} 
            return losses, detections

        



