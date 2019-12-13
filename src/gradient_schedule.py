import logging
from collections import namedtuple

import torch.distributed as dist
import torch

Phase = namedtuple("Phase", ["loss_threshold_next", "gradient_param_prefixes"])

MA_ALPHA = .2
SCHEDULE = [
    (2.5, ["module.roi_heads.box_predictor"]),
    (1.9, ["module.roi_heads.box_predictor", "module.rpn.head.cls_logits", "module.rpn.head.bbox_pred"]),
    (1.5, ["module.roi_heads.box_predictor", "module.rpn.head.cls_logits", "module.rpn.head.bbox_pred", "module.roi_heads.box_head.fc7"]),
    (0., ["module.roi_heads.box_predictor", "module.rpn.head.cls_logits", "module.rpn.head.bbox_pred", "module.roi_heads.box_head"]),
]

class GradientSchedule(object):

    def __init__(self, model, schedule=SCHEDULE):
        self._model = model
        self._schedule = schedule
        self._phase_idx = 0
        self._ma_loss = None
    
        self._activate_gradients()

    def update(self, loss):
        if self._ma_loss is None:
            self._ma_loss = torch.tensor(loss*2)  # We muliply to avoid a lucky first loss triggering the next phase...
        else:
            self._ma_loss = MA_ALPHA*loss + (1-MA_ALPHA)*self._ma_loss

        # Syncronize
        dist.all_reduce(self._ma_loss, op=dist.ReduceOp.SUM)
        self._ma_loss /= dist.get_world_size()

        loss_threshold_next = self._schedule[self._phase_idx][0]
        if self._ma_loss < loss_threshold_next:
            self._phase_idx += 1
            self._activate_gradients()

        nb_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return {"Schedule/ma_loss": self._ma_loss.item(), "Schedule/phase": self._phase_idx, "Scheduler/nb_train_params": nb_params}

    def _activate_gradients(self):
            logging.info("Gradient schedule activated phase %s", self._phase_idx)
            gradient_param_prefixes = self._schedule[self._phase_idx][1]
            for name, parameter in self._model.named_parameters():
                if any(name.startswith(prefix) for prefix in gradient_param_prefixes):
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)
            self.print_trainable_parameters()

    def print_trainable_parameters(self):
        nb_params = 0
        logging.info("Trainable parameters:")
        for name, parameter in self._model.named_parameters():
            if parameter.requires_grad:
                nb_params += parameter.numel()
                logging.info("\t%s (%s)", name, parameter.numel())
    
        logging.info("Model has {:,d} trainable parameters".format(nb_params))

