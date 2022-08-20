import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from enhanced_mmfrra.FCNet import FCNet
from utils import init_weights_with_xavier_uniform


# Top Down Attention Map
class AttnMap(pl.LightningModule):
    '''
    implementation of top down attention
    '''

    def __init__(self, configs):
        super(AttnMap, self).__init__()
        # self.configs = configs

        # conv_channels = configs.PROJECTION_DIM # or configs.PROJECTION_DIM
        self.nonlinear_q_projection = FCNet([configs.HIDDEN_SIZE, configs.PROJECTION_DIM], dropout = configs.DROPOUT_R)
        self.nonlinear_v_projection = FCNet([configs.VISUAL_FEATURES_DIM, configs.PROJECTION_DIM],
                                            dropout = configs.DROPOUT_R)

        # self.linear_p_projection = FCNet([configs.PROJECTION_DIM, configs.VISUAL_FEATURES_DIM], configs.DROPOUT_R)
        # self.linear_q_projection = nn.Linear(configs.HIDDEN_SIZE , configs.PROJECTION_DIM) #256 -> 3072
        # self.linear_v_projection = nn.Linear( configs.VISUAL_FEATURES_DIM, configs.PROJECTION_DIM) #2048 -> 3072
        #@TODO: use weight_norm for all linears!
        self.linear_p_projection = weight_norm(nn.Linear(configs.PROJECTION_DIM,configs.VISUAL_FEATURES_DIM))  # torch.Size([3072,2048])

        init_weights_with_xavier_uniform(self)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def logits(self, q, visual_features):
        """
        :param q: [:, 20, 1024]
        :param visual_features: [:, 100, 2024]
        :return:
        """
        num_objs = visual_features.size(1)  # q:
        # Q = q.unsqueeze(1).repeat(1, num_objs, 1) #(1,100,1)
        Q = q.unsqueeze(1).expand(-1, num_objs, -1)  # (1,100,1)

        # formula (11) z_i = P^T (H^T v_i o G^T Q)
        H = self.nonlinear_v_projection(visual_features)
        G = self.nonlinear_q_projection(Q)

        return self.linear_p_projection(H * G)

    def forward(self, q, visual_features):
        """
        :param q:           question features                   ([:, 1024])
        :param v_reg:           visual & geometric features concat    ([:, 100, 2052])
        :param visual_features:   visual features
        :return:
        """

        z = self.logits(q, visual_features)  # formula (11)

        w = nn.functional.softmax(z, 1)  # formula (10) w_i
        return w


# Attended Joint Map
class TDA(nn.Module):
    def __init__(self, configs):
        super(TDA, self).__init__()

        self.configs = configs

        self.v_att = AttnMap(configs)  # v*f

        self.q_net = FCNet([configs.HIDDEN_SIZE, configs.HIDDEN_SIZE], self.configs.DROPOUT_R)
        self.v_net = FCNet([configs.VISUAL_FEATURES_DIM, configs.HIDDEN_SIZE], self.configs.DROPOUT_R)

        # self.q_net =nn.Linear(configs.HIDDEN_SIZE, configs.HIDDEN_SIZE) # q * f
        # self.v_net = nn.Linear(configs.VISUAL_FEATURES_DIM, configs.HIDDEN_SIZE) # v * f
        # self.vr_net = nn.Linear(configs.VISUAL_FEATURES_DIM, configs.HIDDEN_SIZE)  # v*f

        init_weights_with_xavier_uniform(self)

    def forward(self, q, visual_features):
        """
        :param q:                   question features                   (:,1024)
        :param v_reg:               region veatures (visual+geometric)  (:,100,2052)
        :param visual_features:     visual features                     (:,100,2048)
        :return:
        """

        v_att = self.v_att(q, visual_features)  # formula (11)
        atted_v = (v_att * visual_features).sum(1)  # formula (12) V_att = A^T . V

        # formula (13) (W_r V_vr o W_a W_att) o W_q Q
        # formula (14) (W_r V_vr o W_a W_att) + W_q Q

        q_repr = self.q_net(q)
        v_repr = self.v_net(atted_v)

        # joint_repr = v_repr * q_repr
        return v_repr, q_repr
        # return joint_repr  # torch.Size([256, 1024])
