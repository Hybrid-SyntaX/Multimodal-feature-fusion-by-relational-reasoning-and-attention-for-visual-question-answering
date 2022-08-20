import json
import math
import os
import pickle
import random
import timeit
from collections import OrderedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm


def show_image(image):
    plt.imshow(image.moveaxis(0, -1))
    plt.show()


def split_by_question_type(questions, percentage=1.0, split_type='random'):
    qeustions_per_types = {}
    for question in questions:
        qfi = question['question_family_index']
        if qfi not in qeustions_per_types.keys():
            qeustions_per_types[qfi] = []
        qeustions_per_types[qfi].append(question)

    qeustions_per_types_subset = {}
    qeustions_per_types_subset_list = []
    for k, v in qeustions_per_types.items():
        if k not in qeustions_per_types.keys():
            qeustions_per_types_subset[k] = []

        subset_len = math.floor(len(v) * percentage)
        if split_type == 'start':
            subset_per_q_list = v[0:subset_len]
        elif split_type == 'end':
            subset_per_q_list = v[-subset_len:]
        elif split_type == 'random':
            subset_per_q_list = random.sample(v, subset_len)

        qeustions_per_types_subset[k] = subset_per_q_list
        qeustions_per_types_subset_list.extend(subset_per_q_list)
    return qeustions_per_types_subset_list


def create_word_embedding(vocabs, spacy_model):
    word_embedding = []
    spacy_tool = spacy_model.load()
    for word in vocabs:
        word_embedding.append(spacy_tool(word).vector)
    word_embedding = np.array(word_embedding)
    return word_embedding


def save_h5(content, filename, compression=None, **kwargs):
    var_name = os.path.splitext(os.path.basename(filename))[0]  # filename.split('.')[0]
    if not os.path.exists(filename):
        with h5py.File(filename, 'w') as file:
            file.create_dataset(var_name, data = content, compression = compression, *kwargs)


def load_h5(filename):
    var_name = os.path.splitext(os.path.basename(filename))[0]  # filename.split('.')[0]
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with h5py.File(filename, 'r') as h5_file:
            return np.array(h5_file.get(var_name))




def save_json(data, filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump(data, file)


def load_json(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r') as file:
            return jsVocab.on.load(file)

def save_pickle(vocab, filename):
    if not os.path.exists(filename):
        with open(filename, 'wb') as file:
            pickle.dump(vocab, file)


def load_pickle(filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'rb') as file:
            cached = pickle.load(file)
            return cached

def save_vocabs(vocab, filename):
    if not os.path.exists(filename):
        with open(filename, 'wb') as file:
            pickle.dump(vocab, file)


def load_yaml_as_struct(filename):
    assert os.path.exists(filename) and os.path.getsize(filename) > 0

    with open(filename, 'r') as configs_file:
        configs_yml = yaml.load(configs_file, Loader = yaml.FullLoader)

    return Struct(configs_yml)


class Struct(object):
    def __init__(self, d):
        self.__dict__ = d


#
# class Struct:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)
def run_experiment(func, *args, **kwargs):
    results = OrderedDict({})
    subset = kwargs.get('subset')
    del kwargs['subset']

    start_time = timeit.default_timer()

    output = func(**kwargs)
    exec_time = timeit.default_timer() - start_time

    if isinstance(output, list):
        results[f'{subset}_loss'] = output[0]['test_loss']
        results[f'{subset}_acc'] = output[0]['test_acc']
        results[f'{subset}_precision'] = output[0]['test_precision']
        results[f'{subset}_recall'] = output[0]['test_recall']
        results[f'{subset}_f1'] = output[0]['test_f1']
    if func.__name__ == 'fit':
        results[f'train_time'] = exec_time
    else:
        results[f'{subset}_inference_time'] = exec_time

    return results


def is_tensor_list_empty(tensor_list):
    return tensor_list is not None and len(tensor_list) == 0


def is_tensor_list_nonempty(tensor_list):
    return tensor_list is not None and len(tensor_list) > 0


def init_weights_with_xavier_uniform(model):
    for m in model.modules():  # model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


def save_results(results, filepath):
    for k, v in results.items():
        if isinstance(v, list) or isinstance(v, set):
            results[k] = list(v)

    filepath_csv = os.path.join(filepath, 'results.csv')
    filepath_tsv = os.path.join(filepath, 'results.tsv')
    filepath_json = os.path.join(filepath, 'results.json')
    if not os.path.exists(filepath_json):
        with open(filepath_json, 'w') as log_file:
            json.dump([results], log_file)
    else:
        with open(filepath_json) as log_file:
            log_file_json = json.load(log_file)
        with open(filepath_json, 'w') as log_file:
            log_file_json.append(results)
            json.dump(log_file_json, log_file)

    json_to_csv(filepath_json, filepath_csv, sep = ';')
    json_to_csv(filepath_json, filepath_tsv, sep = '\t')


def json_to_csv(filepath_json, filepath_csv, sep):
    with open(filepath_json) as log_file:
        log_file_json = json.load(log_file)
        with open(filepath_csv, 'w', encoding = 'utf-8') as log_file_csv:
            columns = log_file_json[-1].keys()
            log_file_csv.writelines([sep.join(columns), '\n'])
            for item in log_file_json:
                for c in columns:
                    val = str(item[c]) if c in item else ''
                    log_file_csv.write(val + sep)
                log_file_csv.write('\n')


def balanced_sample(dataset, sample_percentage, categories, split_type='random', val_test_split=False):
    if sample_percentage is None:
        return

    # qeustions_per_types_subset = {}
    indices = []
    odd_indices = []
    even_indices = []
    for k, v in tqdm(categories.items(),
                     desc = f'Subsetting dataset ({sample_percentage}p-{split_type}): Flattening the list'):

        if sample_percentage != 100:
            subset_len = math.floor(len(v) * sample_percentage)
        else:
            subset_len = len(v)
        subset_len_remainder = len(v) - subset_len
        if split_type == 'start':
            subset_per_q_list = v[0:subset_len]
        elif split_type == 'end':
            subset_per_q_list = v[-subset_len:]
        elif split_type == 'random':
            subset_per_q_list = random.sample(v, subset_len)

        # Val/Test split
        if val_test_split:
            odd = subset_per_q_list[1::2]
            even = subset_per_q_list[0::2]
            odd_indices.extend(odd)
            even_indices.extend(even)

        indices.extend(subset_per_q_list)

    # indices = zip(qeustions_per_types_subset.values())
    if val_test_split:
        return Subset(dataset, even_indices), Subset(dataset, odd_indices), even_indices, odd_indices
    return Subset(dataset, indices), indices


def balanced_split(dataset, categories, indices):
    # if sum(lengths) != len(dataset):  # type: ignore
    #     raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    # subset_len = math.floor(len(v) * split_percentage)
    odd_indices = []
    even_indices = []

    # for idx in indices:
    for k, v in tqdm(categories.items(), desc = f'Splitting dataset'):
        for i, item in enumerate(v):
            if item in indices:
                if i % 2 == 0:
                    even_indices.append(item)
                else:
                    odd_indices.append(item)

    # assert len(even_indices)+len(odd_indices) == len(dataset)
    if len(even_indices) + len(odd_indices) != len(dataset):  # type: ignore
        raise ValueError("Odd and even lengths does not equal the length of the input dataset!")

    return Subset(dataset, even_indices), Subset(dataset, odd_indices)
    # offset = 0
    # for length in lengths:
    #     offset = offset + length
    #
    #     indices = torch.arange(offset - length, offset)
    #     yield Subset(dataset, indices)


def caregorize_questions(dataset):
    qeustions_per_types = {}
    dataset.blind = True
    for i, question in enumerate(tqdm(dataset, desc = f'Categorizing questions')):
        qfi = question[-1]  # question_family_index
        if qfi not in qeustions_per_types.keys():
            qeustions_per_types[qfi] = []
        # qeustions_per_types[qfi].append(question)
        qeustions_per_types[qfi].append(i)
    dataset.blind = False
    return qeustions_per_types


def nonrandom_split(dataset, lengths):
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    offset = 0
    for length in lengths:
        offset = offset + length

        indices = torch.arange(offset - length, offset)
        yield Subset(dataset, indices)


def are_dicts_equal(a, b):
    results = []
    for k, v in a.items():
        results.append(b[k] == v if k in b else False)

    return results
