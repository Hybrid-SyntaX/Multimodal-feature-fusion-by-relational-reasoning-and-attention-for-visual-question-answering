import os
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset
from torchtext.vocab import Vocab
from tqdm import tqdm
import numpy as np
import utils
from datamodules.BasicVQADataModule import BasicVQADataModule
from datasets.CLEVRDataset import CLEVRDataset
from datasets.GQADataset import GQADataset


class GQABalancedDataModule(BasicVQADataModule):
    #     'gqa': {
    #         'FRCN_FEAT_SIZE': (100, 2048),
    #         'GRID_FEAT_SIZE': (49, 2048),
    #         'BBOX_FEAT_SIZE': (100, 5),
    @property
    def GRID_FEAT_SIZE(self):

        image_feats, _, _= next(iter(self.train_dataloader()))
        _spatial_features, _,_ = image_feats
        if len(_spatial_features) > 0:
            return _spatial_features[-1].shape
        else: return None

    @property
    def BBOX_FEAT_SIZE(self):
        image_feats, _, _ = next(iter(self.train_dataloader()))
        _, _, _object_bbox_features = image_feats
        if len(_object_bbox_features) > 0 :
            return _object_bbox_features[-1].shape
        else: return None

    @property
    def FRCN_FEAT_SIZE(self):
        image_feats, _, _ = next(iter(self.train_dataloader()))
        _, _object_frcnn_features,_ = image_feats
        if len(_object_frcnn_features) > 0:
            return _object_frcnn_features[-1].shape
        else: return None

    def prepare_data(self,subset='full', *args, **kwargs):

        self.val_data = GQADataset(
            questions_filepath = os.path.join(
                self.questions_dir, 'trimmed_val_balanced_questions.json'),
            object_features_path = os.path.join(self.images_dir, 'objects', 'val_balanced'),
            # spatial_features_path = os.path.join(self.images_dir,'spatial',  'val_balanced') ,
            blind = False
        )
        self.test_data  = GQADataset(
            questions_filepath = os.path.join(self.questions_dir, 'trimmed_testdev_balanced_questions.json'),
            object_features_path = os.path.join(self.images_dir,'objects', 'testdev_balanced'),
            # spatial_features_path =os.path.join(self.images_dir,'spatial', 'testdev_balanced'),
            blind = False)
        self.train_data  = GQADataset(
            questions_filepath = os.path.join(self.questions_dir,'trimmed_train_balanced_questions.json' ),
            object_features_path = os.path.join(self.images_dir,'objects', 'train_balanced'),
            # spatial_features_path = os.path.join(self.images_dir,'spatial', 'train_balanced'),
            blind = False)

        all_data=ConcatDataset([self.train_data,self.val_data,self.train_data])


        self._create_or_cache_vocabs_and_embeddings(all_data, f'gqa_balanced_questions_vocab.pkl',
                                                    f'gqa_balanced_answers_vocab.pkl',
                                                    f'gqa_balanced_word_embedding.h5',
                                                    full_answer_cache_filename=f'gqa_balanced_full_answers_vocab.pkl')

        super().prepare_data(subset)

    @staticmethod
    def proc_bbox_feat(bbox, img_shape):

        # todo: take proc_bbox_feat out of dataset
        bbox_feat = np.zeros((bbox.shape[0], 4), dtype = np.float32)
        HEIGHT, WIDTH = 0, 1
        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[WIDTH])  # width
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[HEIGHT])  # height
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[WIDTH])  # width
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[HEIGHT])  # height
        # (width - x) * (height - y) / (object height * object width)
        #TODO: disable following for mfffraaa
        #bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat

    @staticmethod
    def proc_img_feat(img_feat, img_feat_pad_size):
        # does mostly nothing

        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode = 'constant',
            constant_values = 0
        )

        return img_feat

    def _collate_batch(self, batch):
        spatial_feature_list, question_list, answer_list, fullAnswer_list ,object_frcnn_feature_list, object_bbox_feature_list= [], [], [], [],[],[]
        PAD_IDX,BOS_IDX,EOS_IDX,SPC_IDX = self.questions_vocab['<pad>'],self.questions_vocab['<bos>'],self.questions_vocab['<eos>'],self.answers_vocab['<spc>']
        #
        for _spatial_features, _object_frcnn_features, _object_bbox_features, _question, _answer, _fullAnswer,_metadata in batch:
            #_spatial_features=self.proc_img_feat(_spatial_features,100)
            if _spatial_features is not None:
                spatial_feature_list.append(_spatial_features) #todo: does this need processing?

            if _object_bbox_features is not None and _object_frcnn_features is not None:
                _object_bbox_features = self.proc_img_feat(self.proc_bbox_feat(_object_bbox_features,(_metadata['object']['height'],
                                                                                                      _metadata['object']['width'])),100)
                _object_frcnn_features = self.proc_img_feat(_object_frcnn_features,100)

                object_frcnn_feature_list.append(_object_frcnn_features)
                object_bbox_feature_list.append(_object_bbox_features)

            processed_question = torch.tensor(self.question_pipeline(_question), dtype = torch.int64)
            processed_question=torch.cat([torch.tensor([BOS_IDX]), processed_question,torch.tensor([EOS_IDX])], dim = 0)
            question_list.append(processed_question)

            _answer = _answer.replace(' ', '<spc>') #todo find a better solution
            processed_answer = torch.tensor(self.answer_pipeline(_answer), dtype = torch.int64)
            answer_list.append(processed_answer)

            processed_fullAnswer = torch.tensor(self.fullAnswer_pipeline(_fullAnswer), dtype = torch.int64)
            processed_fullAnswer=torch.cat([torch.tensor([BOS_IDX]), processed_fullAnswer,torch.tensor([EOS_IDX])], dim = 0)
            fullAnswer_list.append(processed_fullAnswer)

            #yield _spatial_features, _object_frcnn_features, _object_bbox_features, processed_question, processed_answer.view(-1), processed_fullAnswer

        if len(spatial_feature_list)>0:
            spatial_feature_list = torch.tensor(spatial_feature_list)  # , dtype = torch.int64
        object_frcnn_feature_list = torch.tensor(object_frcnn_feature_list)  # , dtype = torch.int64
        object_bbox_feature_list = torch.Tensor(object_bbox_feature_list)

        question_list = pad_sequence(question_list, padding_value = PAD_IDX)
        answer_list = torch.tensor(answer_list)
        fullAnswer_list = pad_sequence(fullAnswer_list, padding_value = PAD_IDX)

        question_list = torch.transpose(question_list, 0, 1)
        answer_list = answer_list.view(-1)

        return (spatial_feature_list, object_frcnn_feature_list, object_bbox_feature_list), question_list, (answer_list, fullAnswer_list)

    def _create_vocabs(self, dataset):
        questions_counter = Counter()
        fullAnswers_counter = Counter()
        answers_counter = Counter()
        for _, _, _, question, answer, fullAnswer,metadata in tqdm(dataset,desc = 'Creating vocabs'):
            questions_counter.update(self._tokenizer(question))
            fullAnswers_counter.update(self._tokenizer(fullAnswer))
            answers_counter.update(self._tokenizer(answer))

        questions_vocab = Vocab(questions_counter, specials = self._specials)  # , min_freq = 1
        fullAnswers_vocab = Vocab(fullAnswers_counter, specials = self._specials)
        answers_vocab = Vocab(answers_counter, specials =  self._specials)  # ,

        return questions_vocab, answers_vocab, fullAnswers_vocab