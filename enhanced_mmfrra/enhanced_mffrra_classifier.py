import torch
import torch.nn as nn
import yaml
from torch.nn.utils import weight_norm

import utils
from common_modules.EnhancedCLEVRCNNFeatureExtractor import EnhancedCLEVRCNNFeatureExtractor

from configs.ModelConfig import ModelConfig
from enhanced_mmfrra.enhanced_mffrra_module import MMFRRAModule
from enhanced_mmfrra.tda import TDA
import pytorch_lightning as pl


class EnhancedMMFRRAClassifier(pl.LightningModule):
    def __init__(self, pretrained_emb, answer_size, GRID_FEAT_SIZE=None, FRCN_FEAT_SIZE=None, BBOX_FEAT_SIZE=None,
                 configs_filename=None):
        super(EnhancedMMFRRAClassifier, self).__init__()
        # self.configs = configs
        if not configs_filename:
            configs_filename = 'enhanced_mmfrra/configs.yml'
        self.configs = ModelConfig(GRID_FEAT_SIZE = GRID_FEAT_SIZE,
                                   FRCN_FEAT_SIZE = FRCN_FEAT_SIZE,
                                   BBOX_FEAT_SIZE = BBOX_FEAT_SIZE,
                                   configs_filename = configs_filename)


        question_size =pretrained_emb.shape[0]
        #self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_emb),freeze = False)  # question_tokens * 300
        self.embedding = nn.Embedding(question_size,self.configs.WORD_EMBED_SIZE) #clevr
        if self.configs.USE_IMAGE_FEATURE_EXTRACTOR:
            self.cnn = EnhancedCLEVRCNNFeatureExtractor(self.configs)
            GRID_FEAT_SIZE = (8 * 8, 128)

        self.configs= MMFRRAModule.initialize(self.configs, GRID_FEAT_SIZE, FRCN_FEAT_SIZE, BBOX_FEAT_SIZE)
        self.configs.EPS = torch.finfo(torch.float16).eps
        self.rnn = nn.GRU(
            input_size = self.configs.WORD_EMBED_SIZE,
            hidden_size = self.configs.HIDDEN_SIZE,
            num_layers = 1,
            batch_first = True
        )  # 300 * d_q = 300 * 1024

        self.backbone = TDA(self.configs)

        # Classification layers
        layers = [
            weight_norm(nn.Linear(self.configs.HIDDEN_SIZE, self.configs.FLAT_OUT_SIZE)),
            nn.ReLU(),
            # nn.BatchNorm1d( self.configs.FLAT_OUT_SIZE),
            nn.Dropout(self.configs.CLASSIFER_DROPOUT_R, inplace = True),
            weight_norm(nn.Linear(self.configs.FLAT_OUT_SIZE, answer_size))
        ]

        self.classifer = nn.Sequential(*layers)
        self.mmfraa = MMFRRAModule(self.configs)

        utils.init_weights_with_xavier_uniform(self)
        self.save_hyperparameters()

    def forward(self, questions=None, visual_features=None, geometric_features=None, spatial_features=None):
        # def forward(self, image_features=None, questions=None):  # , frcn_feat, grid_feat, bbox_feat, ques_ix
        if self.configs.USE_IMAGE_FEATURE_EXTRACTOR:
            spatial_features = self.cnn(spatial_features)

        # spatial_features, visual_features, geometric_features = image_features

        lang_feat = self.embedding(questions)
        lang_feat, _ = self.rnn(lang_feat)


        visual_features, v_reg = MMFRRAModule.preprocess_features(spatial_features=spatial_features,
                                                                  visual_features=visual_features,
                                                                  geometric_features = geometric_features)

        q = lang_feat[:, -1]
        # joint_feat = self.backbone(
        #     q = q,
        #     visual_features = visual_features)
        v_repr, q_repr = self.backbone(
            q = q,
            visual_features = visual_features)

        vr_repr = self.mmfraa(q, v_reg, visual_features)  # formula (2)-(10)

        #joint_feat = joint_feat * vr_repr
        joint_feat= (vr_repr * v_repr) * q_repr
        # Classification layers
        proj_feat = self.classifer(joint_feat)

        # return torch.sigmoid(proj_feat) #formula (15)
        # return torch.sigmoid(proj_feat)
        return proj_feat
