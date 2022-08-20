import math
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms

import utils
from datamodules.BasicVQADataModule import BasicVQADataModule
from datasets.CLEVRDataset import CLEVRDataset
from utils import nonrandom_split


class CLEVRHumansDataModule(BasicVQADataModule):

    @property
    def GRID_FEAT_SIZE(self):
        image, _, _ = self.train_data[-1]
        # utils.show_image(image)
        return image.shape
        # return self._grid_feat_size

    @property
    def BBOX_FEAT_SIZE(self):
        pass

    @property
    def FRCN_FEAT_SIZE(self):
        pass

    def prepare_data(self, subset='full', *args, **kwargs):
        # if not self.use_pretrained_features:
        #     image_transforms = transforms.Compose([
        #         #transforms.ToTensor(),
        #                                           # transforms.Pad((0, 80)), # for original size
        #                                          #  #transforms.Pad((0,21)),
        #                                            #transforms.Resize((128, 128)),
        #                                           # transforms.Normalize(0.5, 0.5)
        #                                            ])
        # else:
        image_transforms = None
        _0_05_rad = 0.05 * (180 / math.pi)
        train_transforms = transforms.Compose([#transforms.ToPILImage(),
                                               transforms.Resize((128, 128)),
                                               transforms.Pad(8),
                                               transforms.RandomCrop((128, 128)),
                                               transforms.RandomRotation(_0_05_rad),  # .05 rad
                                               transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                              transforms.ToTensor()])

        val_data = CLEVRDataset(
            questions_filepath = os.path.join(
                self.questions_dir, 'CLEVR-Humans-val.json'),
            images_dir = os.path.join(self.images_dir, 'val'),
            use_cached_image_features = self.use_pretrained_features,
            image_transform = train_transforms)

        # self.test_data = CLEVRDataset(
        #     questions_filepath = os.path.join(self.questions_dir, 'CLEVR-Humans-test.json'),
        #     images_dir = os.path.join(self.images_dir, 'test'),
        #     use_cached_image_features = self.use_pretrained_features)
        self.train_data = CLEVRDataset(
            questions_filepath = os.path.join(self.questions_dir, 'CLEVR-Humans-train.json'),
            images_dir = os.path.join(self.images_dir, 'train'),
            use_cached_image_features = self.use_pretrained_features,
            image_transform = train_transforms
        )

        # self.test_data=self.val_data

        all_data = ConcatDataset([self.train_data, val_data])

        val_len = int(len(val_data) * 0.5)
        test_len = int(len(val_data) - val_len)
        self.val_data, self.test_data = nonrandom_split(val_data, [val_len, test_len])
        # self.test_data=self.val_data

        # assert torch.equal(self.val_data,self.test_data) is False
        self._create_or_cache_vocabs_and_embeddings(all_data, f'clevr_humans_questions_vocab.pkl',
                                                    f'clevr_humans_answers_vocab.pkl',

                                                    f'clevr_humans_word_embedding.h5')

        # image, _, _ = self.train_data[-1]
        # self._grid_feat_size=image.shape

        super().prepare_data(subset)
