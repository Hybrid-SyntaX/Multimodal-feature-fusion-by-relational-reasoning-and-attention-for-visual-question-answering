import json
import os
import timeit
from collections import defaultdict
import random
import math

import torch
from torch.utils.data import Subset
from tqdm import tqdm

import utils
from datasets.CLEVRDataset import CLEVRDataset


# from utils import split_by_question_type


class BalancedClassSampler:
    def __init__(self):
        self.questions_dir = '/mnt/ssd2/Datasets/CLEVR/CLEVR_v1.0/questions'
        self.images_dir = '/mnt/ssd2/Datasets/CLEVR/CLEVR_v1.0/images'
        self.vocabs_dir = '/mnt/ssd2/Datasets/CLEVR/vocabs'
        self.questions = None

    def main(self):
        val_data = CLEVRDataset(
            questions_filepath=os.path.join(
                self.questions_dir, 'CLEVR_val_questions.json'),
            images_dir=os.path.join(self.images_dir, 'val'),
            use_cached_image_features=False,
            image_transform=None, blind=False)
        train_data = CLEVRDataset(
            questions_filepath=os.path.join(
                self.questions_dir, 'CLEVR_train_questions.json'),
            images_dir=os.path.join(self.images_dir, 'val'),
            use_cached_image_features=False,
            image_transform=None, blind=False)

        train_categories_cache_filename = 'clevr_train_categories.json'
        val_categories_cache_filename = 'clevr_val_categories.json'
        train_data_cateogires = utils.load_json(os.path.join(
            self.vocabs_dir, train_categories_cache_filename))
        val_data_cateogires = utils.load_json(os.path.join(
            self.vocabs_dir, val_categories_cache_filename))

        # CREATING VAL/TEST DATASET
        # for sample_percentage in [0.1, 0.25, 0.50, 0.75]:
        sample_percentage = 100
        # train_data_sample, train_indices = utils.balanced_sample(train_data,
        #                                                          sample_percentage = sample_percentage,
        #                                                          categories = train_data_cateogires,
        #                                                          split_type = 'random',
        #                                                          val_test_split = False)
        val_data_sample, test_data_sample, val_indices, test_indices = utils.balanced_sample(val_data,
                                                                                             sample_percentage=sample_percentage,
                                                                                             categories=val_data_cateogires,
                                                                                             split_type='random',
                                                                                             val_test_split=True)

        #utils.save_pickle(train_indices, os.path.join(self.vocabs_dir, f'CLEVR_train_indices_{sample_percentage*100}p_random.pkl'))
        utils.save_pickle(val_indices, os.path.join(
            self.vocabs_dir, f'CLEVR_val_indices_{sample_percentage*100}p_random.pkl'))
        utils.save_pickle(test_indices, os.path.join(
            self.vocabs_dir, f'CLEVR_test_indices_{sample_percentage*100}p_random.pkl'))
        print("Tamam!")
        # Split vaL_data into val and test

        # aaa = balanced_split(val_data,percentage = 0.1)
        # print(len(list(aaa)))
        # print(len(val_data))
        # questions_filepath = questions_filepath = os.path.join(self.questions_dir, 'CLEVR_val_questions.json')
        # with open(questions_filepath, 'r') as questions_file:
        #     self.questions = json.load(questions_file)['questions']

        # start_split = balanced_split(val_data, percentage = 0.1)
        # end_split = balanced_split(val_data, percentage = 0.1, split_type = 'end')
        # rand_split = balanced_split(val_data, percentage = 0.1, split_type = 'random')
    # torch_split = balanced_split(val_data, percentage = 0.1)
    # print(len(list(start_split)))
    # print(len(list(end_split)))
    # print(len(list(rand_split)))
    # print(len(list(torch_split)))
    # for rnd in rand_split:
    #   print(rnd)

    # print('loi')
    # Categorizing questions

    # print(qeustions_per_types_subset.keys())
    # for k,v in qeustions_per_types_subset.items():
    #     print(k,': ',len(v))
    # print(qeustions_per_types_subset[0])


if __name__ == '__main__':
    # print("KIR")
    BalancedClassSampler().main()
