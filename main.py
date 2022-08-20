import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from sys import platform
from glob import glob
import pytorch_lightning as pl
import torch
import torch.autograd
import yaml
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profiler import SimpleProfiler

from VQAClassifier import VQAClassifier
from butd.BottomUpTopDownClassifier import BottomUpTopDownClassifier
from butd_with_mmfrra.BottomUpTopDownClassifierWithMMFRRA import BottomUpTopDownClassifierWithMMFRRA
from cnn_rnn.CNNRNNClassifier import CNNRNNClassifier
from configs.TrainingConfig import TrainingConfig
from datamodules.CLEVRDataModule import CLEVRDataModule
from datamodules.CLEVRHumansDataModule import CLEVRHumansDataModule
from datamodules.COCOQADataModule import COCOQADataModule
from datamodules.GQABalancedDataModule import GQABalancedDataModule
from enhanced_mmfrra.enhanced_mffrra_classifier import EnhancedMMFRRAClassifier
from mmfrra.mffrra_classifier import MMFRRAClassifier
from utils import run_experiment, save_results  # , CheckpointEveryNSteps

datamodules = {'clevr-humans': CLEVRHumansDataModule,
               'clevr': CLEVRDataModule,
               'coco-qa': COCOQADataModule,
               'gqa-balanced': GQABalancedDataModule}
vqa_systems = {'butd-mmfrra': VQAClassifier,
               'butd': VQAClassifier,
               'mmfrra': VQAClassifier,
               'enhanced-mmfrra': VQAClassifier,
               'cnn-rnn': VQAClassifier
               }
models = {'butd-mmfrra': BottomUpTopDownClassifierWithMMFRRA,
          'mmfrra': MMFRRAClassifier,
          'enhanced-mmfrra': EnhancedMMFRRAClassifier,
          'butd': BottomUpTopDownClassifier,
          'cnn-rnn': CNNRNNClassifier}


def save_grad_history(trainer: pl.Trainer, model, *args):
    vars(model.autoclip)
    pass


class SaveGradHistoryCallback(Callback):
    # def on_load_checkpoint(
    #         self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', callback_state: Dict[str, Any]
    # ) -> None:
    #     pass
    #
    # def on_init_start(self, trainer: 'pl.Trainer') -> None:
    #     pass

    # def on_save_checkpoint(
    #         self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', checkpoint: Dict[str, Any]
    # ) -> dict:
    #     epoch = checkpoint['epoch']
    #     step = checkpoint['global_step']

    def on_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        epoch = trainer.current_epoch
        filepath = os.path.join(trainer.logger.log_dir, f'auto_clip_data_epoch={epoch}.json')
        auto_clip_data = vars(pl_module.autoclip)
        with open(filepath, 'w') as file:
            json.dump(auto_clip_data, file)
    # def on_init_start(self, trainer):
    #     print('Starting to init trainer!')
    #
    # def on_init_end(self, trainer):
    #     print('trainer is init now')
    #
    # def on_train_end(self, trainer, pl_module):
    #     print('do something when training ends')


class Experiment:
    def __init__(self, args):
        self.args = args
        self.results = OrderedDict({'experiment_name':self.args.name,
                                    'model_name': self.args.model,
                                    'dataset': self.args.dataset,
                                    'subset':self.args.dataset_subset,
                                    'datetime': datetime.now().isoformat(),
                                    'tr_acc': 0,
                                    'val_acc': 0,
                                    'test_acc': 0,
                                    'train_loss': 0,
                                    'val_loss': 0,
                                    'test_loss': 0,
                                    'train_time': 0,
                                    'train_inference_time': 0,
                                    'val_inference_time': 0,
                                    'test_inference_time': 0,
                                    'experiment_type': '',
                                    'original_checkpoint_path': None,
                                    'comments': self.args.comments
                                    })
        if self.args.train:
            self.results['experiment_type'] = 'resume + train' if self.args.checkpoint else 'train'
        if self.args.test:
            self.results['experiment_type'] = 'test'
        if self.args.checkpoint:
            self.results['original_checkpoint_path'] = self.args.checkpoint

        self.dm, self.classifier, self.trainer, self.vqa_system, self.logger = self.initialize(args.dataset, args.model)

    def experiment_name(self):
        return f'{self.args.model}_on_{self.args.dataset}'

    def test(self):
        # assert checkpoint and os.path.exists(checkpoint)
        # assert os.path.exists(checkpoint)
        if self.args.checkpoint and os.path.exists(self.args.checkpoint) \
                and 'resume' not in self.results['experiment_type']:
            print('Loading checkpoint ', self.args.checkpoint)
            training_configs = TrainingConfig(
                os.path.join(os.path.dirname(os.path.dirname(self.args.checkpoint)), 'hparams.yaml'))
            print('training configs', vars(training_configs))
            self.vqa_system = vqa_systems[self.args.model].load_from_checkpoint(self.args.checkpoint,
                                                                                model = self.classifier,
                                                                                training_configs = training_configs)
        else:
            print(f'No checkpoint was found at the path "{self.args.checkpoint}" or no model was trained')

        # model.eval()

        # self.test_and_log(checkpoint, dataset, dm, logger, model, trainer, vqa_system = the_vqa_system,
        #                   train_config = the_vqa_system.training_configs)

        # self.results.update(train_results)
        self.results.update(run_experiment(func = self.trainer.test,
                                           test_dataloaders = self.dm.test_dataloader(),
                                           model = self.vqa_system,
                                           subset = 'test'))

        self.results.update(run_experiment(func = self.trainer.test,
                                           test_dataloaders = self.dm.val_dataloader(),
                                           model = self.vqa_system,
                                           subset = 'val'))

        # print('train data')

        self.results.update(run_experiment(func = self.trainer.test,
                                           test_dataloaders = self.dm.train_dataloader(
                                               shuffle = False),
                                           model = self.vqa_system,
                                           subset = 'tr'))
        if self.args.train:
            self.results['experiment_type'] = ' + '.join([self.results['experiment_type'], 'test'])
        self.results.move_to_end('comments')

        # self.logger.log_metrics(self.results)

        save_results(self.results, self.logger.save_dir)
        self.logger.save()

    def train(self):
        print('vqa system configs (training configs)', vars(self.vqa_system.training_configs))

        self.results.update(run_experiment(func = self.trainer.fit,
                                           model = self.vqa_system,
                                           datamodule = self.dm,
                                           subset = 'train'))

        # Logging
        self.logger.log_hyperparams(self.vqa_system.all_configs)
        self.logger.save()
        self.results.update(self.vqa_system.all_configs)
        with open(os.path.join(self.logger.log_dir, 'model_summary.txt'), 'w') as model_summary:
            print(self.vqa_system.model, file = model_summary)
        # Testing
        # self.test_and_log(self.logger.log_dir, results = results, train_config = self.vqa_system.training_configs)

        self.test()

        print(f'Model {self.logger.log_dir} checkpointed successfully')

        if self.args.dataset_subset != 'single-batch':
            with open("resume.sh","w") as resume_sh:
                checkpoint_path = os.path.join(self.logger.log_dir,'checkpoints')
                checkpoint_path = glob(f"{checkpoint_path}/last*")[0] if len(glob(f"{checkpoint_path}/last*")) ==1 else ""
                resume_sh.write((f'python main.py --train --model {self.args.model} --dataset {self.args.dataset}'
                                 f' --checkpoint "{checkpoint_path}" --name "{self.args.name}" '
                                 f' --comments "{self.args.comments}" --dataset_subset "{self.args.dataset_subset}"'))

    def find_lr(self):
        # dm, classifier, trainer, vqa_model, logger = self.initialize(dataset, model_name)  # model_conifg

        # trainer = pl.Trainer()

        # Run learning rate finder
        lr_finder = self.trainer.tuner.lr_find(self.vqa_system, datamodule = self.dm)

        # Results can be found in
        results = lr_finder.results

        # Plot with
        # fig = lr_finder.plot(suggest = True)
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        self.vqa_system.hparams.lr = new_lr
        self.vqa_system.training_configs.auto_lr = new_lr

        # Fit model
        # trainer.fit(vqa_model)

        return results

    def initialize(self, dataset, model):
        dev_mode_ds_percentage = 0.05
        training_configs = TrainingConfig(self.args.training_configs)  # TrainingConfig('configs/training_configs.yml')
        # model_conifg = configs[model]()
        use_pinmemory = self.args.gpus > 0

        use_pretrained_features = False
        if use_pretrained_features:
            img_featres = self.args.features_dir
            print('USING PRETRAINED FEATURES at ', img_featres)
        else:
            img_featres = self.args.images_dir
            print('USING RAW IMAGE  at ', img_featres)
        dm = datamodules[dataset](questions_dir = self.args.questions_dir,
                                  vocabs_dir = self.args.vocabs_dir,
                                  images_dir = img_featres,
                                  batch_size = training_configs.BATCH_SIZE,
                                  use_pretrained_features = use_pretrained_features,
                                  num_workers = training_configs.NUM_WORKERS,
                                  pin_memory = use_pinmemory)

        dm.prepare_data(self.args.dataset_subset)
        dm.setup()
        training_configs.IMAGE_FEATURE_TYPES = dm.IMAGE_FEATURE_TYPES
        training_configs.NUM_CLASSES = dm.answer_dim
        if self.args.dev_mode:
            training_configs.TRAIN_DATA_LEN = len(dm.train_data)* dev_mode_ds_percentage
        else:
            training_configs.TRAIN_DATA_LEN = len(dm.train_data)
        # training_configs.PRECISION = 16 if self.args.gpus > 0 else 32
        training_configs.PRECISION = 32
        if training_configs.PRECISION == 16:
            training_configs.EPS = torch.finfo(torch.float16).eps
        else:
            training_configs.EPS = torch.finfo(torch.float32).eps

        training_configs.QUESTION_DIM = dm.question_dim
        classifier = models[model](pretrained_emb = dm.word_embedding,
                                   answer_size = dm.answer_dim,
                                   GRID_FEAT_SIZE = dm.GRID_FEAT_SIZE,
                                   FRCN_FEAT_SIZE = dm.FRCN_FEAT_SIZE,
                                   BBOX_FEAT_SIZE = dm.BBOX_FEAT_SIZE
                                   # ,configs_filename=model_conifg
                                   )

        logger = CSVLogger(self.args.logs_dir, name = f'{model}_on_{dataset}')
        # logger.log_hyperparams(training_configs)
        if self.args.max_epochs:
            training_configs.MAX_EPOCHS = self.args.max_epochs
        if self.args.checkpoint:
            checkpoint = self.args.checkpoint
            assert os.path.exists(checkpoint)
        else:
            checkpoint = None
        vqa_system = vqa_systems[model](classifier, training_configs)
        if not os.path.exists(logger.log_dir):
            os.makedirs(logger.log_dir)
        profiler = SimpleProfiler(output_filename = os.path.join(logger.log_dir, 'performance_profile.logs'))
        
        _best_model_checkpoint = ModelCheckpoint(
            verbose = True,
            dirpath = os.path.join(logger.log_dir, 'checkpoints'),
            monitor = 'val_loss',
            filename = 'best_{epoch:02d}-{step}-{val_loss:.2f}-{val_acc:.2f}',
            save_top_k = 2,
            mode = 'min',
            save_last = True,
            every_n_val_epochs =1
            )
        _step_model_checkpoint = ModelCheckpoint(
            verbose = False,
            dirpath = os.path.join(logger.log_dir, 'checkpoints'),
            filename = 'step_{epoch:02d}-{step}-{val_loss:.2f}-{val_acc:.2f}',
            # save_last = True,
            every_n_train_steps = 10
        )
        # input(len(dm.train_data)//10)
        _best_model_checkpoint.CHECKPOINT_NAME_LAST = 'last_{epoch:02d}-{step}-{val_loss:.2f}-{val_acc:.2f}'
        # _step_model_checkpoint.CHECKPOINT_NAME_LAST = 'last_{epoch:02d}-{step}-{valid_loss:.2f}-{valid_accuracy:.2f}'
        # fff =len(dm.train_data)/2
        # mid_epoch_checkpoint = CheckpointEveryNSteps(save_step_frequency=fff)
        # _last_model_checkpoint=ModeldatasetCheckpoint(
        #     save_last = True,
        #     #filename = '{epoch:02d}-{valid_loss:.2f}',
        #     filename='epoch={epoch:02d}-step={step}-valid_loss={valid_loss:.2f}'
        # )
        lr_monitor = LearningRateMonitor(logging_interval = 'epoch')
        # _save_grad_history=LambdaCallback(on_save_checkpoint=save_grad_history)
        callbacks = [_step_model_checkpoint,
                     _best_model_checkpoint]

        if training_configs.AUTO_GRAD_CLIP:
            callbacks.append(SaveGradHistoryCallback())

        trainer = pl.Trainer.from_argparse_args(self.args,
                                                gpus = self.args.gpus,
                                                precision = training_configs.PRECISION,
                                                max_epochs = training_configs.MAX_EPOCHS,
                                                # gradient_clip_val = training_configs.GRAD_NORM_CLIP,
                                                # gradient_clip_algorithm = 'norm',
                                                accumulate_grad_batches = training_configs.GRAD_ACCU_STEPS,
                                                logger = logger,
                                                resume_from_checkpoint = checkpoint,
                                                profiler = profiler,
                                                auto_lr_find = 'lr' if training_configs.AUTO_LR_FIND else False,
                                                callbacks = callbacks,
                                                stochastic_weight_avg = training_configs.STOCHASTIC_WEIGHT_AVG,
                                                weights_summary = 'full',
                                                #
                                                # limit_train_batches = 0.05,
                                                # limit_test_batches = 0.05,
                                                # limit_val_batches = 0.05,
                                                # limit_predict_batches = 0.05,
                                                # log_every_n_steps = 1,

                                                # track_grad_norm = 2,

                                                # overfit_batches =0.0001,
                                                # num_sanity_val_steps = 0
                                                # fast_dev_run=1,

                                                )  # num_sanity_val_steps=0
        # self.args.detect_anomaly=True

        if args.dev_mode:
            #torch.autograd.set_detect_anomaly(True)
            trainer.limit_train_batches =dev_mode_ds_percentage
            trainer.limit_test_batches =dev_mode_ds_percentage
            trainer.limit_val_batches =dev_mode_ds_percentage
            trainer.limit_predict_batches = dev_mode_ds_percentage
            trainer.log_every_n_steps = 1
        # , num_sanity_val_steps=0
        self.results['checkpoint_path'] = logger.log_dir
        self.results.move_to_end('checkpoint_path')

        return dm, classifier, trainer, vqa_system, logger  # ,model_conifg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type = int, default = 1)

    if platform.startswith('linux'):
        with open('configs/datasets_linux.yaml') as f:
            datasets_yaml = yaml.load(f, Loader = yaml.FullLoader)
    else:
        with open('configs/datasets_windows.yaml') as f:
            datasets_yaml = yaml.load(f, Loader = yaml.FullLoader)

    parser.add_argument('--comments', type = str, default = '')
    parser.add_argument('--name', type = str, default = '')

    parser.add_argument('--results_file', type = str, default = os.path.join(datasets_yaml['logs-dir'], 'results.csv'))
    parser.add_argument('--dataset_subset', type = str, default = 'full')
    parser.add_argument('--training_configs', type = str, default = 'configs/training_configs.yml')
    parser.add_argument('--logs_dir', type = str, default = datasets_yaml['logs-dir'])

    parser.add_argument('--max_epochs', type = int)
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--grad_accu_steps', type = int)

    parser.add_argument('--checkpoint', type = str, )
    parser.add_argument('--save_checkpoint', type = str, )

    parser.add_argument('--test', action = 'store_true')
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--find_lr', action = 'store_true')

    parser.add_argument('--model', type = str, choices = models.keys(),required = True)
    parser.add_argument('--dataset', type = str, choices = datamodules.keys(),required = True)

    parser.add_argument('--dev_mode', action = 'store_true')

    # parser.add_argument('--max_epochs', type = int)
    # parser.add_argument('--batch_size', type = int)
    # parser.add_argument('--grad_accu_steps', type = int)
    # parser.add_argument('--max_epochs', type = int)

    args = parser.parse_args()

    # args.num_sanity_val_steps=0
    print(datasets_yaml[args.dataset])
    # TODO: use another variable for these
    # if not args.questions_dir:
    args.questions_dir = datasets_yaml[args.dataset]['questions_dir']
    # if not args.images_dir:
    args.images_dir = datasets_yaml[args.dataset]['images_dir']
    # if not args.features_dir:
    args.features_dir = datasets_yaml[args.dataset]['features_dir']
    # if not args.vocabs_dir:
    args.vocabs_dir = datasets_yaml[args.dataset]['vocabs_dir']

    assert os.path.exists(args.questions_dir)
    assert os.path.exists(args.images_dir)
    # assert os.path.exists(args.vocabs_dir)
    if not os.path.exists(args.vocabs_dir):
        os.makedirs(args.vocabs_dir)

    print('==== Command line arguments ====\n', args, '\n====================================\n')
    my_vqa = Experiment(args)
    if args.train:
        print(f'Training {args.model} on {args.dataset}')
        my_vqa.train()  # model = args.model, dataset = args.dataset
    elif args.test:
        assert args.checkpoint
        print(f'Testing {args.model} on {args.dataset} from {args.checkpoint}')
        # my_vqa.test(model = args.model, dataset = args.dataset, checkpoint = args.checkpoint)
        my_vqa.test()
    elif args.find_lr:
        print(f'Finding LR for {args.model} on {args.dataset}')
        print(my_vqa.find_lr())
    else:
        print('You must specify an action! \n --train \n --test \n --find_lr\n')
