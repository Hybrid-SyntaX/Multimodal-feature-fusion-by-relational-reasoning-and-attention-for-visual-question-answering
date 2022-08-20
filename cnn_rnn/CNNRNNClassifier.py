import torch
import torch.nn as nn
import yaml
from torch.nn.utils.weight_norm import weight_norm

import utils
from butd.tda import TDA
import pytorch_lightning as pl

from configs.ModelConfig import ModelConfig
from enhanced_mmfrra.FCNet import FCNet
from utils import Struct


class CNNRNNClassifier(pl.LightningModule):
    def __init__(self, pretrained_emb, answer_size, GRID_FEAT_SIZE=None, FRCN_FEAT_SIZE=None, BBOX_FEAT_SIZE=None,
                 configs_filename=None):
        super(CNNRNNClassifier, self).__init__()

        # with open('butd/configs.yml','r') as configs_file:
        #     self.configs_yml = yaml.load(configs_file,Loader = yaml.FullLoader)
        # self.configs = configs
        # self.configs =  utils.load_yaml_as_struct('butd/configs.yml')
        if not configs_filename:
            configs_filename = 'cnn_rnn/configs.yml'
        self.configs = ModelConfig(GRID_FEAT_SIZE = GRID_FEAT_SIZE,
                                   FRCN_FEAT_SIZE = FRCN_FEAT_SIZE,
                                   BBOX_FEAT_SIZE = BBOX_FEAT_SIZE,
                                   configs_filename = configs_filename)

        # self.data_configs = utils.load_yaml_as_struct('configs/datasets_linux.yaml')
        # with open('configs/datasets_linux.yaml') as f:
        #     self.data_configs =Struct(yaml.load(f,Loader = yaml.FullLoader)['clevr-humans'])
        # self.data_configs = self.data_configs['clevr-humans']
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_emb),
                                                      freeze = False)

        self.rnn = nn.LSTM(
            input_size = self.configs.WORD_EMBED_SIZE,
            hidden_size = self.configs.HIDDEN_SIZE,
            num_layers = 1,
            batch_first = True
        )

        if FRCN_FEAT_SIZE and BBOX_FEAT_SIZE:
            self.USE_BBOX_FEAT = True
            self.bbox_linear, self.frcn_linear = self.gqa_init()
        if GRID_FEAT_SIZE:
            self.grid_linear = self.clevr_init()
        # self.backbone = TDA(self.configs)
        # self.image_nonlinear = FCNet([self.configs.IMG_FEAT_SIZE, self.configs.HIDDEN_SIZE], self.configs.DROPOUT_R)
        # self.image_nonlinear = nn.Linear(self.configs.IMG_FEAT_SIZE, answer_size)  # 1024

        self.nonlinear = FCNet([self.configs.IMG_FEAT_SIZE * 2, self.configs.HIDDEN_SIZE], self.configs.DROPOUT_R)
        self.linear = weight_norm(nn.Linear(self.configs.HIDDEN_SIZE, 1), dim = None)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(196,
                                  self.configs.FLAT_OUT_SIZE), dim = None),
            nn.ReLU(),
            # nn.BatchNorm1d(self.configs.FLAT_OUT_SIZE),
            nn.Dropout(self.configs.CLASSIFER_DROPOUT_R, inplace = True),
            weight_norm(nn.Linear(self.configs.FLAT_OUT_SIZE, answer_size), dim = None)  # output
        ]

        self.classifer = nn.Sequential(*layers)
        self.save_hyperparameters()

    def clevr_init(self):
        # return  nn.Linear(self.configs.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], self.configs.HIDDEN_SIZE)
        return nn.Linear(self.configs.GRID_FEAT_SIZE[1], self.configs.HIDDEN_SIZE)

    def gqa_init(self):
        # imgfeat_linear_size = int(self.configs.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]) #2048

        imgfeat_linear_size = int(self.configs.FRCN_FEAT_SIZE[1])  # 2048

        if self.USE_BBOX_FEAT:
            bbox_linear = nn.Linear(4, self.configs.BBOXFEAT_EMB_SIZE)  # 5 , 1024
            imgfeat_linear_size += self.configs.BBOXFEAT_EMB_SIZE

        # if self.configs.USE_AUX_FEAT:
        #     grid_linear = nn.Linear(
        #         self.configs.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], self.configs.HIDDEN_SIZE)

        frcn_linear = nn.Linear(imgfeat_linear_size, self.configs.HIDDEN_SIZE)  # 2048, 512

        return bbox_linear, frcn_linear

    def clevr_forward(self, image_features):
        grid_feat = image_features

        img_feat = self.grid_linear(grid_feat)

        return img_feat

    def gqa_forward(self, frcn_feat, bbox_feat):  # grid_feat
        # img_feat_mask = make_mask(frcn_feat)

        if self.configs.USE_BBOX_FEAT:
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim = -1)

        img_feat = self.frcn_linear(frcn_feat)

        return img_feat  # , img_feat_mask

    def forward(self, questions=None, spatial_features=None, visual_features=None,
                geometric_features=None):  # , frcn_feat, grid_feat, bbox_feat, ques_ix
        # def forward(self, image_features=None,questions=None):  # , frcn_feat, grid_feat, bbox_feat, ques_ix
        #     spatial_features,visual_features,geometric_features = image_features
        # frcn_feat=torch.empty([256, 100, 2048]),bbox_feat=torch.empty([256, 100, 5]),
        # ques_ix=torch.empty([256, 20],dtype = torch.int)
        # print(frcn_feat.shape,bbox_feat.shape,ques_ix.shape)
        # exit()
        # torch.empty([256, 100, 2048])
        # torch.empty([256, 100, 5])
        # torch.empty([256, 20])
        # Language features
        lang_feat = self.embedding(questions)
        lang_feat, _ = self.rnn(lang_feat)
        q = lang_feat[:, -1]
        # Image features

        if utils.is_tensor_list_nonempty(visual_features) and \
                utils.is_tensor_list_nonempty(geometric_features):
            img_feat = self.gqa_forward(visual_features, geometric_features)

        if utils.is_tensor_list_nonempty(spatial_features):
            img_feat = self.clevr_forward(spatial_features)

        # img_feat=self.image_nonlinear(img_feat)
        # img_feat=img_feat.view(img_feat.size(0), img_feat.size(1), -1)
        # num_objs = img_feat.size(1)  # q:
        # Q = q.unsqueeze(1).repeat(1, num_objs, 1) #(1,100,1)
        q = q.unsqueeze(1).expand(-1, img_feat.size(1), -1)

        vq = torch.cat((q, img_feat), dim = 2)
        vq_nonlinear = self.nonlinear(vq)
        logits = self.linear(vq_nonlinear).squeeze()
        #joint_feat = nn.functional.softmax(logits, 1)
        # fusion fed into Backbone Framework
        # joint_feat = self.backbone(
        #     lang_feat[:, -1],
        #     img_feat)

        # Classification layers
        proj_feat = self.classifer(logits)  # torch.Size([10, 196, 1024])

        return proj_feat  # torch.Size([10, 196, 26])
