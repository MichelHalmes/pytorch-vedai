
## DISTRIBUTED ##
NB_PROCESSES = 6

## MODEL ##
IMAGE_SIZE = 1024
BATCH_SIZE = 1
EVAL_STEPS = 25
CHECKPOINT_DIR = "./data/model"
CHECKPOINT_NAME = "model.pth.tar"
LOG_DIR = "./data/logs"

## DATASET ##
DATA_PATH = "./data/sets/{name}"
IMAGES_PATH = "images/{id_}.jpg"
ANNOTATIONS_PATH = "annotations/{id_}.txt"
EVALSET_PCT = .1
CROP_TO_SIZE = (1024, 1024)  # H, W

## EVAL ##
PLOT_DETECTIONS_MIN_SCORE=.07


## GRADIENT SCHEDULES ##
INIT_SCHEDULE = [
    (1000, ["module.roi_heads.box_predictor", "module.rpn"]),
    (500, ["module.roi_heads.box_predictor", "module.rpn", "module.roi_heads.box_head.fc7"]),
    (500, ["module.roi_heads", "module.rpn"]),
    (1000, ["module.roi_heads", "module.rpn", "module.backbone.fpn"]),
    (500, ["module.roi_heads", "module.rpn", "module.backbone.fpn", "module.backbone.body.layer4"]),
    (1000, ["module.roi_heads", "module.rpn", "module.backbone.fpn", "module.backbone.body.layer4", "module.backbone.body.layer3"]),
    (1000, ["module.roi_heads", "module.rpn", "module.backbone.fpn", "module.backbone.body.layer4", "module.backbone.body.layer3", "module.backbone.body.layer2"]),
]

TRAINED_SCHEDULE = [
    (800, ["module.roi_heads.box_predictor", "module.rpn.head.cls_logits", "module.rpn.head.bbox_pred"]),
    (500, ["module.roi_heads.box_predictor", "module.rpn"]),
    (500, ["module.roi_heads.box_predictor", "module.rpn", "module.roi_heads.box_head.fc7"]),
    (500, ["module.roi_heads", "module.rpn"]),
    (1000, ["module.roi_heads", "module.rpn", "module.backbone.fpn"]),
    (500, ["module.roi_heads", "module.rpn", "module.backbone.fpn", "module.backbone.body.layer4"]),
    (1000, ["module.roi_heads", "module.rpn", "module.backbone.fpn", "module.backbone.body.layer4", "module.backbone.body.layer3"]),
    (1000, ["module.roi_heads", "module.rpn", "module.backbone.fpn", "module.backbone.body.layer4", "module.backbone.body.layer3", "module.backbone.body.layer2"]),
]
