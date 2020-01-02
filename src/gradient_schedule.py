import logging
from collections import namedtuple

import torch.distributed as dist
import torch

class GradientSchedule(object):

    def __init__(self, model, schedule):
        self._model = model
        self._schedule = schedule
        self._phase_idx = 0

        self._activate_gradients()

    def is_done(self):
        return self._phase_idx >= len(self._schedule)

    def update(self, step):
        prev_phase_idx = self._phase_idx
        cumm_total_steps = 0
        self._phase_idx = 0
        for nb_steps, _ in self._schedule:
            cumm_total_steps += nb_steps
            if step < cumm_total_steps:
                # We found the current phase we schould be in
                break
            self._phase_idx += 1

        if prev_phase_idx != self._phase_idx:
            self._activate_gradients()

        nb_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return {"Schedule/phase": self._phase_idx, "Scheduler/nb_train_params": nb_params}

    def _activate_gradients(self):
        if self.is_done():
            return 

        logging.info("Gradient schedule activated phase %s", self._phase_idx)
        gradient_param_prefixes = self._schedule[self._phase_idx][1]
        for name, parameter in self._model.named_parameters():
            if any(name.startswith(prefix) for prefix in gradient_param_prefixes):
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        if dist.get_rank() != 0:
            return

        nb_params = 0
        logging.info("Trainable parameters:")
        for name, parameter in self._model.named_parameters():
            if parameter.requires_grad:
                nb_params += parameter.numel()
                logging.info(f"\t{name} ({parameter.numel():,d})")
    
        logging.info(f"Model has {nb_params:,d} trainable parameters")

