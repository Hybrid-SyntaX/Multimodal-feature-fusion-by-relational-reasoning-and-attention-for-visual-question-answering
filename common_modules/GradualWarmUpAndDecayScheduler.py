import warnings
from collections import OrderedDict
from typing import Optional, Callable

import pytorch_lightning as pl
import torch
from torch import nn, optim
import torchmetrics

# import wandb
# except: pass


# from pytorch_model_summary  import summary
from torch.optim import Optimizer
# from torch.optim.lr_scheduler import _LRScheduler # OneCycleLR,
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmUpAndDecay(_LRScheduler):
    def __init__(self, optimizer,
                 warmup_epochs,
                 warmup_step,
                 decay_epoch,
                 decay_step,
                 decay_rate,
                 last_epoch=-1,
                 verbose=False):
        self.warmup_epochs = warmup_epochs
        self.warmup_step = warmup_step
        self.decay_epoch = decay_epoch
        self.decay_step = decay_step
        self.decay_rate = decay_rate

        super(GradualWarmUpAndDecay, self).__init__(optimizer, last_epoch, verbose)

    def gradual_warmup_and_decay(self, current_lr):
        # todo : turn gradual_warmup_and_decay into scheduler
        epoch = self.last_epoch
        if 0 <= epoch < self.warmup_epochs:
            lr = current_lr + self.warmup_step
        # elif epoch > self.decay_epoch and self.decay_step is 'even' and epoch % 2 == 0:
        elif epoch > self.decay_epoch and epoch % self.decay_step == 0:
            lr = current_lr * self.decay_rate
        else:
            lr = current_lr

        return lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:  # or (self.last_epoch % self.step_size != 0):
            #return [group['lr'] for group in self.optimizer.param_groups]
            return self.base_lrs
        else:
            return [self.gradual_warmup_and_decay(group['lr'])
                    for group in self.optimizer.param_groups]
