import math
import os

import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms

import utils
from datamodules.BasicVQADataModule import BasicVQADataModule
from datasets.CLEVRDataset import CLEVRDataset
from utils import nonrandom_split


class CLEVRDataModule(BasicVQADataModule):

    @property
    def FRCN_FEAT_SIZE(self):
        pass

    @property
    def BBOX_FEAT_SIZE(self):
        pass

    @property
    def GRID_FEAT_SIZE(self):
        print("LENGH : ", len(self.train_data))
        image, _, _, _ = self.train_data[-1]

        return image.shape

    def prepare_data(self, subset='full', *args, **kwargs):

        # if not self.use_pretrained_features:
        #     image_transforms = transforms.Compose([
        #                                         transforms.ToTensor(),
        #                                            #transforms.Pad((0, 80)),
        #
        #                                            #transforms.Pad((0, 21)),
        #
        #                                            #transforms.Resize((128, 128)),
        #                                            #transforms.Normalize(0.5, 0.5)
        #                                            ])
        # else:
        _0_05_rad = 0.05 * (180 / math.pi)  # .05 rad
        train_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Pad(8),
            transforms.RandomCrop((128, 128)),
            transforms.RandomRotation(_0_05_rad),
            transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                              transforms.ToTensor()])

        val_data = CLEVRDataset(
            questions_filepath = os.path.join(
                self.questions_dir, 'CLEVR_val_questions.json'),
            images_dir = os.path.join(self.images_dir, 'val'),
            use_cached_image_features = self.use_pretrained_features,
            image_transform = train_transforms, blind = False)
        # self.test_data = CLEVRDataset(
        #     questions_filepath = os.path.join(self.questions_dir, 'CLEVR_test_questions.json'),
        #     images_dir = os.path.join(self.images_dir, 'test'),
        #     use_cached_image_features = self.use_pretrained_features)
        self.train_data = CLEVRDataset(
            questions_filepath = os.path.join(self.questions_dir, 'CLEVR_train_questions.json'),
            images_dir = os.path.join(self.images_dir, 'train'),
            use_cached_image_features = self.use_pretrained_features,
            image_transform = train_transforms,
            blind = False)

        all_data = ConcatDataset([self.train_data, val_data])

        self._create_or_cache_vocabs_and_embeddings(all_data, f'clevr_questions_vocab.pkl',
                                                    f'clevr_answers_vocab.pkl',
                                                    f'clevr_word_embedding.h5')


        if subset =='full' or subset =='single-batch':
            #val_len = int(len(val_data) * 0.5)
            #test_len = int(len(val_data) - val_len)
            #self.val_data, self.test_data = nonrandom_split(val_data, [val_len, test_len])

            #balanced test/val
            val_indices = utils.load_pickle(os.path.join(self.vocabs_dir, f'CLEVR_val_indices_100p_random.pkl'))
            test_indices = utils.load_pickle(os.path.join(self.vocabs_dir, f'CLEVR_test_indices_100p_random.pkl'))
            assert val_indices and test_indices

            self.val_data = Subset(val_data, val_indices)
            self.test_data = Subset(val_data, test_indices)
            print('Loading balanced test and val set complete')
        elif subset != 'single-batch':
            train_indices = utils.load_pickle(os.path.join(self.vocabs_dir,f'CLEVR_train_indices_{subset}.pkl'))
            val_indices = utils.load_pickle(os.path.join(self.vocabs_dir,f'CLEVR_val_indices_{subset}.pkl'))
            test_indices = utils.load_pickle(os.path.join(self.vocabs_dir,f'CLEVR_test_indices_{subset}.pkl'))

            if train_indices and val_indices and test_indices:
                self.train_data = Subset(self.train_data, train_indices)
                self.val_data= Subset(val_data,val_indices)
                self.test_data = Subset(val_data, test_indices)
            else:
                raise Exception('Subset not found!')
        # self.val_data, self.test_data = utils.balanced_split(val_data, val_data_cateogires, indices)


        super().prepare_data(subset)
