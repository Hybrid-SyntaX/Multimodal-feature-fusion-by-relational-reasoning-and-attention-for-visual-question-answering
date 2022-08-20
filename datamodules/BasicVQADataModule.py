import os
import string
from abc import abstractmethod
from collections import Counter, OrderedDict
from typing import Optional
import spacy
import en_core_web_lg as spacy_model
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Subset
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm

import utils
from datasets.CLEVRDataset import CLEVRDataset


class BasicVQADataModule(pl.LightningDataModule):
    def __init__(self, questions_dir, vocabs_dir, images_dir, batch_size: int = 8, num_workers=8,
                 use_pretrained_features=False, pin_memory=True):
        super(BasicVQADataModule).__init__()
        self.use_pretrained_features = use_pretrained_features
        self.images_dir = images_dir
        self.train_data, self.val_data, self.test_data = None, None, None
        self.questions_dir = questions_dir

        self.batch_size = batch_size
        self.vocabs_dir = vocabs_dir

        self._tokenizer = get_tokenizer('basic_english')

        # self.question_pipeline = lambda x: [self.questions_vocab[token] for token in self._tokenizer(x)]
        # self.answer_pipeline = lambda x: [self.answers_vocab[token] for token in self._tokenizer(x)]
        # self.fullAnswer_pipeline = lambda x: [self.fullAnswers_vocab[token] for token in self._tokenizer(x)]

        self.questions_vocab, self.answers_vocab, self.fullAnswers_vocab = None, None, None
        self.word_embedding = None
        self._specials = ['<unk>', '<pad>', '<bos>', '<eos>', '<spc>']
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self._has_teardown_fit = False
        self._has_teardown_test = False
        self._has_prepared_data = False
        self._has_setup_fit=False
        self._has_setup_validate=False
        self._has_setup_test = False
        self._log_hyperparams=False
        #
        # self._GRID_FEAT_SIZE=None
        # self._FRCN_FEAT_SIZE=None
        # self._BBOX_FEAT_SIZE=None

    def question_pipeline(self, x):
        return [self.questions_vocab[token] for token in self._tokenizer(x)]

    def answer_pipeline(self, x):
        return [self.answers_vocab[token] for token in self._tokenizer(x)]

    def fullAnswer_pipeline(self, x):
        return [self.fullAnswers_vocab[token] for token in self._tokenizer(x)]

    # @abstractmethod
    def prepare_data(self, subset, *args, **kwargs):
        if subset == 'single-batch':
            self.val_data = Subset(self.val_data, torch.arange(0, self.batch_size))
            self.test_data = Subset(self.test_data, torch.arange(0, self.batch_size))
            self.train_data = Subset(self.train_data, torch.arange(0, self.batch_size))

    @property
    @abstractmethod
    def GRID_FEAT_SIZE(self):
        pass

    @property
    @abstractmethod
    def BBOX_FEAT_SIZE(self):
        pass

    @property
    @abstractmethod
    def FRCN_FEAT_SIZE(self):
        pass

    @property
    def IMAGE_FEATURE_TYPES(self):
        feature_types = []
        if self.GRID_FEAT_SIZE is not None:
            feature_types.append('spatial')
        if self.FRCN_FEAT_SIZE is not None:
            feature_types.append('object')
        if self.BBOX_FEAT_SIZE is not None:
            feature_types.append('bbox')
        if len(feature_types) == 0:
            feature_types.append('raw')
        return set(feature_types)

    @property
    def answer_dim(self):
        return len(self.answers_vocab)

    @property
    def question_dim(self):
        return len(self.questions_vocab)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_length(self):
        return len(self.train_dataloader())

    def val_length(self):
        return len(self.val_dataloader())

    def test_length(self):
        return len(self.test_dataloader())

    def train_dataloader(self, shuffle=True):
        # torch.arange(0,10000)
        # torch.utils.data.SubsetRandomSampler(indices, generator=None)
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size = self.batch_size,
                                           shuffle = shuffle,
                                           collate_fn = self._collate_batch,
                                           drop_last = True,
                                           num_workers = self.num_workers,
                                           pin_memory = self.pin_memory)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = self._collate_batch,
                                           drop_last = True,
                                           num_workers = self.num_workers,
                                           pin_memory = self.pin_memory)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size = self.batch_size,
                                           shuffle = False,
                                           collate_fn = self._collate_batch,
                                           drop_last = True,
                                           num_workers = self.num_workers,
                                           pin_memory = self.pin_memory)

    def _collate_batch(self, batch):
        question_list, answer_list, image_list = [], [], []
        PAD_IDX, BOS_IDX, EOS_IDX, SPC_IDX = self.questions_vocab['<pad>'], self.questions_vocab['<bos>'], \
                                             self.questions_vocab['<eos>'], self.answers_vocab['<spc>']

        for _image, _question, _answer, _qtype in batch:
            #_image=torch.from_numpy(_image).float()
            image_list.append(_image)

            # vocabs = list(self.questions_vocab.stoi.values())[len(self._specials) - 1:]
            # random_question =np.random.choice(vocabs,np.random.randint(10,15))
            # processed_question = torch.tensor(random_question, dtype = torch.int64)
            _question = _question.replace('&', ' and ')
            _question = _question.replace('/', ' or ')
            _question = _question.translate(str.maketrans('', '', string.punctuation)).strip()
            processed_question = torch.tensor(self.question_pipeline(_question), dtype = torch.int64)
            processed_question = torch.cat([torch.tensor([BOS_IDX]), processed_question, torch.tensor([EOS_IDX])],
                                           dim = 0)
            question_list.append(processed_question)

            _answer = _answer.translate(str.maketrans('', '', string.punctuation)).strip()
            _answer = _answer.replace(' ', '<spc>')  # todo find a better solution
            processed_answer = torch.tensor(self.answer_pipeline(_answer), dtype = torch.int64)
            answer_list.append(processed_answer)

        question_list = pad_sequence(question_list, padding_value = PAD_IDX)
        question_list = torch.transpose(question_list, 0, 1)

        answer_list = torch.tensor(answer_list).view(-1)
        # if self.use_pretrained_features:  # TODO: refactor this later
        #     image_list = torch.tensor(image_list)
        # else:
        #     image_list = torch.stack(image_list)
        #image_list = torch.tensor(image_list)
        image_list = torch.stack(image_list)
#
        return image_list, question_list, answer_list

    def _create_vocabs_3(self, dataset):
        questions_counter = Counter()
        answers_counter = Counter()

        for _, question, answer,_ in tqdm(dataset, desc = 'Creating vocabs'):
            question = question.translate(str.maketrans('', '', string.punctuation)).strip()
            questions_counter.update([t.lower() for t in question.split()])
            answers_counter.update([t.lower() for t in answer.split()])

        questions_counter_sorted = sorted(questions_counter.items(), key = lambda x: x[1], reverse = True)
        answers_counter_sorted = sorted(answers_counter.items(), key = lambda x: x[1], reverse = True)

        questions_counter_odict = OrderedDict(questions_counter_sorted)
        answers_counter_odict = OrderedDict(answers_counter_sorted)

        unk_token = '<unk>'
        default_index = -1
        from torchtext.vocab import vocab
        questions_vocab = vocab(questions_counter_odict)  # , min_freq = 1
        answers_vocab = vocab(answers_counter_odict)  # ,, specials = self._specials
        print(answers_vocab)
        print(answers_vocab.get_stoi())

        if unk_token not in questions_vocab:
            questions_vocab.insert_token(unk_token,0)

        if unk_token not in answers_vocab:
            answers_vocab.insert_token(unk_token,0)

        questions_vocab.set_default_index(default_index)
        answers_vocab.set_default_index(default_index)

        return questions_vocab, answers_vocab

    def _create_vocabs(self, dataset):
        from torchtext.vocab import build_vocab_from_iterator

        answers_list=[]
        questions_list=[]

        for _, question, answer,_ in tqdm(dataset, desc = 'Creating vocabs'):
            question = question.translate(str.maketrans('', '', string.punctuation)).strip() #REMOVE STOP WORDS
            questions_list.append([t.lower() for t in question.split()])
            answers_list.append([t.lower() for t in answer.split()])


        questions_vocab = build_vocab_from_iterator(questions_list, specials = self._specials)
        answers_vocab= build_vocab_from_iterator(answers_list, specials =[])

        #print(questions_vocab._parameters)
        #print(vars(questions_vocab.vocab))

        #input()
        unk_token = '<unk>'
        default_index = -1
        if unk_token not in questions_vocab:
            questions_vocab.insert_token(unk_token, 0)
        if unk_token not in answers_vocab:
            answers_vocab.insert_token(unk_token, 0)

        questions_vocab.set_default_index(default_index)
        answers_vocab.set_default_index(default_index)

        questions_vocab.set_default_index(questions_vocab[unk_token])
        answers_vocab.set_default_index(answers_vocab[unk_token])
        #print(questions_vocab.get_stoi())
        return questions_vocab, answers_vocab

    def _create_vocabs_v_0_8_0(self, dataset):
        questions_counter = Counter()
        answers_counter = Counter()

        for _, question, answer,_ in tqdm(dataset, desc = 'Creating vocabs'):
            question = question.translate(str.maketrans('', '', string.punctuation)).strip()
            questions_counter.update(self._tokenizer(question))
            answers_counter.update(self._tokenizer(answer))

        questions_vocab = Vocab(questions_counter, specials = self._specials)  # , min_freq = 1
        answers_vocab = Vocab(answers_counter, specials = [])  # ,, specials = self._specials

        return questions_vocab, answers_vocab

    def _create_or_cache_vocabs_and_embeddings(self, all_data, question_cache_filename, answer_cache_filename,
                                               word_embedding_filename, full_answer_cache_filename=None):
        self.questions_vocab = utils.load_pickle(os.path.join(self.vocabs_dir, question_cache_filename))
        self.answers_vocab = utils.load_pickle(os.path.join(self.vocabs_dir, answer_cache_filename))
        if full_answer_cache_filename:
            self.fullAnswers_vocab = utils.load_pickle(os.path.join(self.vocabs_dir, full_answer_cache_filename))

        if self.questions_vocab is None or self.answers_vocab is None:
            print('Creating vocab cache')
            if full_answer_cache_filename:
                self.questions_vocab, self.answers_vocab, self.fullAnswers_vocab = self._create_vocabs(all_data)
            else:
                self.questions_vocab, self.answers_vocab = self._create_vocabs(all_data)

            #TODO: DO NOT CACHE VOCAB OBJECT
            utils.save_pickle(self.questions_vocab, os.path.join(self.vocabs_dir, question_cache_filename))
            utils.save_pickle(self.answers_vocab, os.path.join(self.vocabs_dir, answer_cache_filename))
            if full_answer_cache_filename:
                utils.save_pickle(self.fullAnswers_vocab, os.path.join(self.vocabs_dir, full_answer_cache_filename))
        else:
            print('Vocabs loaded from cache')

        self.word_embedding = utils.load_h5(os.path.join(self.vocabs_dir, word_embedding_filename))
        if self.word_embedding is None:
            print('Creating embedding cfreqsache')
            #spacy_model = spacy.load("en_core_web_lg")
            self.word_embedding = utils.create_word_embedding(self.questions_vocab.stoi, spacy_model)
            utils.save_h5(self.word_embedding, os.path.join(self.vocabs_dir, word_embedding_filename),
                          compression = 'lzf')
        else:
            print('Embedding loaded from cache')
