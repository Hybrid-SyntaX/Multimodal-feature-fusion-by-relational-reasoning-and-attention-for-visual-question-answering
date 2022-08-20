import os

from torch.utils.data import ConcatDataset

import utils
from datamodules.BasicVQADataModule import BasicVQADataModule
from datasets.CLEVRDataset import CLEVRDataset
from datasets.COCOQADataset import COCOQADataset


class COCOQADataModule(BasicVQADataModule):

    def prepare_data(self):
        self.train_data  = COCOQADataset(
            questions_dir = os.path.join(self.questions_dir, 'train'),
            images_dir = os.path.join(self.images_dir, 'train2014'))

        test_data  = COCOQADataset(
            questions_dir = os.path.join(self.questions_dir, 'test'),
            images_dir = os.path.join(self.images_dir, 'val2014'))

        ds_length = int(len(test_data))
        lengths = [int(ds_length * 0.5), int(ds_length * 0.5)]
        self.val_data, self.test_data = utils.nonrandom_split(dataset = test_data,lengths = lengths)



        all_data=ConcatDataset([self.train_data,self.val_data,self.test_data])

        self._create_or_cache_vocabs_and_embeddings(all_data, f'cocoqa_questions_vocab.pkl',
                                                    f'cocoqa_answers_vocab.pkl',
                                                    f'cocoqa_word_embedding.h5')



