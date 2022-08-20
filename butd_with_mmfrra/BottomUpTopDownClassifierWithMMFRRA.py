import pytorch_lightning as pl
import torch
import torch.nn as nn
from mmfrra.mffrra_module import MMFRRAModule
from torch.nn.utils.weight_norm import weight_norm

import utils
from butd.tda import TDA
from configs.ModelConfig import ModelConfig


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim = -1) == 0).unsqueeze(1).unsqueeze(2)


class BottomUpTopDownClassifierWithMMFRRA(pl.LightningModule):
    def __init__(self, pretrained_emb, answer_size, GRID_FEAT_SIZE=None, FRCN_FEAT_SIZE=None, BBOX_FEAT_SIZE=None,
                 configs_filename=None):
        super(BottomUpTopDownClassifierWithMMFRRA, self).__init__()

        # with open('butd/configs.yml','r') as configs_file:
        #     self.configs_yml = yaml.load(configs_file,Loader = yaml.FullLoader)
        # self.configs = configs
        # self.configs =  utils.load_yaml_as_struct('butd/configs.yml')
        if not configs_filename:
            configs_filename = 'butd_with_mmfrra/configs.yml'
        self.configs = ModelConfig(GRID_FEAT_SIZE = GRID_FEAT_SIZE,
                                   FRCN_FEAT_SIZE = FRCN_FEAT_SIZE,
                                   BBOX_FEAT_SIZE = BBOX_FEAT_SIZE,
                                   configs_filename = configs_filename)

        # self.data_configs = utils.load_yaml_as_struct('configs/datasets_linux.yaml')
        # with open('configs/datasets_linux.yaml') as f:
        #     self.data_configs =Struct(yaml.load(f,Loader = yaml.FullLoader)['clevr-humans'])
        # self.data_configs = self.data_configs['clevr-humans']
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_emb))

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
        self.backbone = TDA(self.configs)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(self.configs.HIDDEN_SIZE,
                                  self.configs.FLAT_OUT_SIZE), dim = None),
            nn.ReLU(),
            nn.Dropout(self.configs.CLASSIFER_DROPOUT_R, inplace = True),
            weight_norm(nn.Linear(self.configs.FLAT_OUT_SIZE, answer_size), dim = None)
        ]

        # MMFRRA part
        self.configs = MMFRRAModule.initialize(self.configs, GRID_FEAT_SIZE, FRCN_FEAT_SIZE, BBOX_FEAT_SIZE)
        # if GRID_FEAT_SIZE:
        #     if len(GRID_FEAT_SIZE)==2:
        #         self.configs.IMG_FEAT_SIZE=GRID_FEAT_SIZE[1]
        #         self.configs.VISUAL_FEATURES_DIM=GRID_FEAT_SIZE[1]
        #     elif len(GRID_FEAT_SIZE)==3:
        #         self.configs.IMG_FEAT_SIZE = GRID_FEAT_SIZE[1]*GRID_FEAT_SIZE[1]
        #         self.configs.VISUAL_FEATURES_DIM = GRID_FEAT_SIZE[1]*GRID_FEAT_SIZE[1]
        # elif FRCN_FEAT_SIZE and BBOX_FEAT_SIZE:
        #     self.configs.IMG_FEAT_SIZE=FRCN_FEAT_SIZE[1]+BBOX_FEAT_SIZE[1]
        #     self.configs.VISUAL_FEATURES_DIM=FRCN_FEAT_SIZE[1]

        self.mmfraa = MMFRRAModule(self.configs)
        # self.vr_net = nn.Linear(self.configs.VISUAL_FEATURES_DIM, self.configs.HIDDEN_SIZE)  # v*f
        # MMFRRA end of part
        self.classifer = nn.Sequential(*layers)
        utils.init_weights_with_xavier_uniform(self)
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

        # Image features
        #
        # if utils.is_tensor_list_nonempty(visual_features) and \
        #         utils.is_tensor_list_nonempty(geometric_features):
        #     img_feat = self.gqa_forward(visual_features, geometric_features)
        #
        # if utils.is_tensor_list_nonempty(spatial_features):
        #     img_feat = self.clevr_forward(spatial_features)

        visual_features, v_reg = MMFRRAModule.preprocess_features(spatial_features = spatial_features,
                                                                  visual_features = visual_features,
                                                                  geometric_features = geometric_features)

        q = lang_feat[:, -1]
        # fusion fed into Backbone Framework
        joint_feat = self.backbone(
            q,
            visual_features)

        vr_repr = self.mmfraa(q, v_reg, visual_features)  # formula (2)-(10)

        joint_feat = joint_feat * vr_repr

        # <MMFRRA>
        # if utils.is_tensor_list_nonempty(spatial_features):
        #     # spatial_features = torch.flatten(spatial_features, start_dim = 2)
        #     v_reg = spatial_features
        #
        #     vr = self.mmfraa(q, v_reg, img_feat)
        #     vr_repr = self.vr_net(vr)
        #     joint_feat = joint_feat * vr_repr
        # # </MMFRRA>
        # Classification layers
        proj_feat = self.classifer(joint_feat)

        return proj_feat
