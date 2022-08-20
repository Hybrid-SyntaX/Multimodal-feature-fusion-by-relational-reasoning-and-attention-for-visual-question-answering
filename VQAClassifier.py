from collections import OrderedDict
from collections import OrderedDict
from typing import Any

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR

from common_modules.AutoClip import AutoClip, compute_grad_norm, compute_weights_norm
from common_modules.GradualWarmUpAndDecayScheduler import GradualWarmUpAndDecay


class VQAClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, training_configs):
        super().__init__()

        self.schedulers = {}
        self.model = model
        print(self.model)
        self.training_configs = training_configs

        self.iter_size = self.training_configs.TRAIN_DATA_LEN // self.training_configs.BATCH_SIZE  # 97
        self.max_iter_size = self.iter_size * self.training_configs.MAX_EPOCHS
        # self.dm=datamodule
        # lr=0
        self.criterion = nn.CrossEntropyLoss(reduction = self.training_configs.LOSS_REDUCTION)

        # nn.BCEWithLogitsLoss
        # if self.configs.AUTO_LR_FIND:
        #     self.lr=lr
        # else:
        self.scheduler = None
        # if self.training_configs.SCHEDULER == 'OneCycleLR':
        #     self.lr = float(self.training_configs.LR_MAX)
        #
        # else:
        #     self.lr = float(self.training_configs.LR_BASE)
        # self.hparams.lr = self.lr  # todo: this worked

        # self.save_hyperparameters()

        self.metric_accuracy = torchmetrics.Accuracy()
        self.metric_recall = torchmetrics.Recall()
        self.metric_precision = torchmetrics.Precision()
        self.metric_f1 = torchmetrics.F1(num_classes = training_configs.NUM_CLASSES)

        if self.training_configs.AUTO_GRAD_CLIP:
            self.automatic_optimization = False
            self.autoclip = AutoClip(percentile = self.training_configs.AUTO_GRAD_CLIP_PERCENTILE)
            print('Using AutoClip...')

    def forward(self, x):
        image_feats, question, answer = x
        if len(answer) == 2:
            answer, _fullAnswer = answer

        if {'object', 'bbox'}.intersection(self.training_configs.IMAGE_FEATURE_TYPES):
            spatial_features, visual_features, geometric_features = image_feats
            output = self.model(question, spatial_features = spatial_features,
                                visual_features = visual_features,
                                geometric_features = geometric_features)
        elif {'spatial'}.intersection(self.training_configs.IMAGE_FEATURE_TYPES):
            output = self.model(question, spatial_features = image_feats)
        else:
            raise Exception(f"Input feature is invalid", {'len(image_feats)': len(image_feats),
                                                          'IMAGE_FEATURE_TYPES': self.training_configs.IMAGE_FEATURE_TYPES})

        return output, answer

    def training_step(self, batch, batch_idx):

        if self.training_configs.AUTO_GRAD_CLIP:
            opt = self.optimizers()
            opt.zero_grad()

        # with torch.autograd.detect_anomaly():
        output, answer = self(batch)

        loss = self.criterion(output, answer)
        y_hat = torch.argmax(output, dim = 1)


        if self.training_configs.AUTO_GRAD_CLIP:
            self.manual_backward(loss)
            grad_norm, clip_value = self.autoclip(self.model)

            opt.step()

            # self.log_lr()
            lr = self.optimizers().param_groups[0]['lr']

            _log_on_step = True
            _log_on_epoch = not _log_on_step
            self.log('lr', lr, prog_bar = True, on_step = _log_on_step, on_epoch = _log_on_epoch)
#            self.log('lr_epoch', lr, prog_bar = True, on_step = False, on_epoch = False,logger = False)
            self.log('wt_n1', compute_weights_norm(self.model, 1), on_step = _log_on_step,
                     on_epoch = _log_on_epoch,
                     prog_bar = False,
                     logger = True)
            self.log('wt_n2', compute_weights_norm(self.model, 2), on_step = _log_on_step,
                     on_epoch = _log_on_epoch,
                     prog_bar = False,
                     logger = True)
            self.log('grad_n1', compute_grad_norm(self.model, 1), on_step = _log_on_step, on_epoch = _log_on_epoch,
                     prog_bar = False,
                     logger = True)
            self.log('grad_n2', grad_norm, on_step = _log_on_step, on_epoch = _log_on_epoch, prog_bar = True,
                     logger = True)
            self.log('clip', clip_value, on_step = _log_on_step, on_epoch = _log_on_epoch, prog_bar = True,
                     logger = True)
            self.log('loss', loss, on_step = _log_on_step, on_epoch = _log_on_epoch, prog_bar = True, logger = True)

            self.log('tr_loss', loss, on_step = False, on_epoch = True)
            self.log('tr_acc', self.metric_accuracy(y_hat, answer), on_step = False, on_epoch = True,
                     prog_bar = True)

        return loss

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def validation_step(self, batch, batch_idx):
        output, answer = self(batch)

        loss = self.criterion(output, answer)
        y_hat = torch.argmax(output, dim = 1)

        # accuracy = self.calculate_accuracy(answer, output)
        self.log('val_loss', loss, on_step = False, on_epoch = True)
        self.log('val_acc', self.metric_accuracy(y_hat, answer), on_step = False,
                 on_epoch = True, prog_bar = True)

        self.log('val_precision', self.metric_precision(y_hat, answer), on_step = False, on_epoch = True)
        self.log('val_recall', self.metric_recall(y_hat, answer), on_step = False, on_epoch = True)
        self.log('val_f1', self.metric_f1(y_hat, answer), on_step = False, on_epoch = True)

        # self.log('valid_accuracy', accuracy)

        return loss

    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     self.log('valid_accuracy_epoch', self.accuracy.compute())

    def test_step(self, batch, batch_idx):
        # image,question,y = batch

        # x = x.view(x.size(0), -1)
        # y_hat= self.model(spatial_features = image, question = question)
        output, answer = self(batch)

        loss = self.criterion(output, answer)
        y_hat = torch.argmax(output, dim = 1)
        # accuracy = self.calculate_accuracy(y, y_hat)

        # Logging to TensorBoard by default
        self.log('test_loss', loss, on_step = False, on_epoch = True)
        self.log('test_acc', self.metric_accuracy(y_hat, answer), on_step = False, on_epoch = True)
        self.log('test_precision', self.metric_precision(y_hat, answer), on_step = False, on_epoch = True)
        self.log('test_recall', self.metric_recall(y_hat, answer), on_step = False, on_epoch = True)
        self.log('test_f1', self.metric_f1(y_hat, answer), on_step = False, on_epoch = True)

        return loss

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    # if self.trainer.current_epoch == 0:
    #     checkpoint['optimizer_states'][0]['param_groups'][0]['lr']= 0.001334
    #     checkpoint['lr_schedulers'][0]['_last_lr'] = 0.001334

    def on_train_epoch_start(self) -> None:
        if self.training_configs.AUTO_GRAD_CLIP:
            self.autoclip.grad_history.clear()

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    #     pass
    #
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # if self.training_configs.SCHEDULER and self.training_configs.AUTO_GRAD_CLIP:  # :
        #if batch_idx < self.iter_size:
        if self.training_configs.SCHEDULER:
            self.lr_schedulers().step()

        # print(batch_idx)

    def configure_optimizers(self):
        # todo: there'sa bug with auto lr, it makes self.hparams.lr=None!
        optimizer = optim.Adamax(params = self.model.parameters(),
                                 betas = self.training_configs.OPT_PARAMS_BETAS,
                                 eps = float(self.training_configs.OPT_PARAMS_EPS),
                                 # eps= self.training_configs.EPS,
                                 lr = self.training_configs.LR_BASE,
                                 weight_decay = float(self.training_configs.WEIGHT_DECAY))  # self.lr

        # optimizer = optim.AdamW(params = self.model.parameters(),
        #                          #betas = self.training_configs.OPT_PARAMS_BETAS,
        #                          #eps = float(self.training_configs.OPT_PARAMS_EPS),
        #                          # eps= self.training_configs.EPS,
        #                          lr = self.training_configs.LR_BASE,
        #                         # weight_decay = float(self.training_configs.WEIGHT_DECAY)
        #                         )

        scheduler_name = self.training_configs.SCHEDULER
        if scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer,
                                   max_lr = self.training_configs.LR_MAX,
                                   epochs = self.training_configs.MAX_EPOCHS,
                                   steps_per_epoch = self.training_configs.BATCH_SIZE,
                                   #total_steps = self.training_configs.TRAIN_DATA_LEN,
                                   )
        elif scheduler_name == 'CyclicLR':
            scheduler = CyclicLR(optimizer,
                                 max_lr = self.training_configs.LR_MAX,
                                 base_lr = self.training_configs.LR_BASE,
                                 step_size_up = self.training_configs.SCHEDULER_PARAMS['STEP_SIZE_MULTIPLIER'] * self.iter_size,
                                 # step_size_up = 10 * self.iter_size,
                                 #                                     step_size_down = 0,
                                 cycle_momentum = False,
                                 mode = self.training_configs.SCHEDULER_PARAMS['MODE']
                                 )
        elif scheduler_name == 'GradualWarmUpAndDecay':
            scheduler = GradualWarmUpAndDecay(optimizer,
                                              warmup_epochs =
                                              self.training_configs.SCHEDULER_PARAMS.get(
                                                  'warmup_epochs'),
                                              warmup_step =
                                              self.training_configs.SCHEDULER_PARAMS.get(
                                                  'warmup_step'),
                                              decay_epoch = self.training_configs.SCHEDULER_PARAMS.get(
                                                  'decay_epoch'),
                                              decay_step = self.training_configs.SCHEDULER_PARAMS.get(
                                                  'decay_step'),
                                              decay_rate =
                                              self.training_configs.SCHEDULER_PARAMS.get('decay_rate'),
                                              verbose = False)

        if scheduler_name:
            return [optimizer], [scheduler]
        else:
            return optimizer

    @property
    def all_configs(self):
        configs = OrderedDict()
        # self.training_configs.LR = self.lr
        configs['lr'] = self.optimizers().param_groups[0]['lr']
        configs['optimizer'] = self.optimizers().optimizer.__class__.__name__
        # configs['scheduler_last_lr'] = self.scheduler.get_last_lr() if self.scheduler else None
        configs['trained_epochs'] = self.trainer.current_epoch
        configs.update(vars(self.model.configs))
        configs.update(vars(self.training_configs))
        configs.update(vars(self.hparams))
        return configs
