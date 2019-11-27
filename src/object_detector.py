
import requests
from PIL import Image

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


class ObjectDetector():
    def __init__(self, num_classes):
        self._model = self._init_pretrained_model(num_classes)

    def _init_pretrained_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)

        box_predictor = FastRcnnBoxPredictor(
            in_channels=model.roi_heads.box_head.fc7.out_features,
            num_classes=num_classes)

        model.roi_heads.box_predictor = box_predictor
        return model

        # url = 'https://images.fineartamerica.com/images-medium-large-5/dog-and-cat-driving-car-through-snowy-john-danielsjohan-de-meester.jpg'
        # response = requests.get(url, stream = True)
        # image = Image.open(response.raw)
        # transf = transforms.ToTensor()
        # img_tensor = transf(image)
        # model.eval()
        # output = model([img_tensor])
        # print(output)




