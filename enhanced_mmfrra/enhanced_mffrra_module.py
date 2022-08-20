import timeit

import torch
from torch import nn

# Rp
import utils
from enhanced_mmfrra.FCNet import FCNet
from utils import init_weights_with_xavier_uniform


def symmetric_softmax(x):
    batch_size = x.shape[0]
    feat_dim = x.shape[2]
    ssm = nn.functional.softmax(x.view(batch_size, 1, -1), 2)
    return ssm.view(batch_size, 1, feat_dim, -1)


''' CHECKD '''


class PairWiseRelation(nn.Module):
    @staticmethod
    def conv2d(input_size, output_size):
        return [nn.Conv2d(in_channels = input_size,
                          out_channels = output_size,
                          kernel_size = (1, 1),
                          stride = (1, 1),
                          padding = (0, 0)),
                nn.MaxPool2d(kernel_size = output_size),
                #nn.AdaptiveMaxPool2d(output_size = output_size),
                #nn.AdaptiveAvgPool2d(output_size),
                #nn.FractionalMaxPool2d(kernel_size = output_size,output_size=1),
                #nn.LPPool2d(norm_type = 2, kernel_size = output_size),  # manhathan very good
                #nn.AvgPool2d(kernel_size = output_size),
                nn.ReLU(),
                nn.InstanceNorm2d(output_size)]

    def __init__(self, configs):
        super(PairWiseRelation, self).__init__()

        layers = []
        input_size = configs.SUBSPACE_DIM
        for _ in range(0, configs.CONV_LAYERS_NUM-1):
            output_size = input_size // 2
            layers.extend(self.conv2d(input_size,output_size))
            input_size = output_size
        layers.extend(self.conv2d(input_size,1))
        layers.append(nn.Dropout(configs.DROPOUT_R, inplace = True))
        self.pg_layers=nn.Sequential(*layers)

        init_weights_with_xavier_uniform(self)

    def forward(self, v_pc):
        """
        :param v_pc:    (:,256,100,100) d_s*k*K
        :return:
        """
        vdot_pc = self.pg_layers(v_pc)
        vdot_pc = torch.add(vdot_pc, vdot_pc.transpose(2, 3))

        return symmetric_softmax(vdot_pc)

        # return nn.functional.softmax(vdot_pc.squeeze(dim = 1), dim = 2)


''' CHECKD '''


class InnerGroupRelation(nn.Module):
    @staticmethod
    def conv2d(input_size, output_size,dilation_size):
        return [nn.Conv2d(in_channels = input_size,
                          out_channels = output_size,
                          kernel_size = (3, 3),
                          stride = (1, 1),
                          dilation = dilation_size,
                          padding = dilation_size),
                #nn.AdaptiveMaxPool2d(output_size = output_size),
                nn.MaxPool2d(kernel_size = output_size),
                #nn.AdaptiveAvgPool2d(output_size),
                #nn.FractionalMaxPool2d(kernel_size = output_size,output_size = 1),
                #nn.LPPool2d(norm_type = 2,kernel_size = output_size), #manhathan
                #nn.AvgPool2d(kernel_size = output_size),
                nn.ReLU(),
                nn.InstanceNorm2d(output_size)]

    def __init__(self, configs):
        super(InnerGroupRelation, self).__init__()

        layers = []
        input_size = configs.SUBSPACE_DIM
        dilation_size = 1
        for _ in range(0, configs.CONV_LAYERS_NUM-1):
            output_size = input_size // 2

            layers.extend(self.conv2d(input_size,output_size,dilation_size))

            input_size = output_size
            dilation_size=dilation_size * 2
        layers.extend(self.conv2d(input_size,1,dilation_size))
        layers.append(nn.Dropout(configs.DROPOUT_R, inplace = True))
        self.rg_layers=nn.Sequential(*layers)

        init_weights_with_xavier_uniform(self)

    def forward(self, x):
        vumlat_pc = self.rg_layers(x)
        vumlat_pc = torch.add(vumlat_pc, vumlat_pc.transpose(2, 3))
        return symmetric_softmax(vumlat_pc)
        # .squeeze()

        # return nn.functional.softmax(vumlat_pc.squeeze(dim = 1), dim = 2)


class MMFRRAModule(nn.Module):
    def __init__(self, configs):
        super(MMFRRAModule, self).__init__()

        # self.linear_q =nn.Linear(configs.HIDDEN_SIZE, configs.SUBSPACE_DIM) # f * d_s
        self.linear_q = FCNet([configs.HIDDEN_SIZE, configs.SUBSPACE_DIM], configs.DROPOUT_R)
        self.linear_v = FCNet([configs.IMG_FEAT_SIZE, configs.SUBSPACE_DIM], configs.DROPOUT_R)
        # self.linear_v =nn.Linear(configs.IMG_FEAT_SIZE, configs.SUBSPACE_DIM) # d_v * d_s

        self.rp = PairWiseRelation(configs)
        self.rg = InnerGroupRelation(configs)
        self.vr_net = FCNet([configs.VISUAL_FEATURES_DIM, configs.HIDDEN_SIZE],
                            configs.DROPOUT_R)  # TODO: inspect thi sone

        init_weights_with_xavier_uniform(self)

    def relational_reasoning(self, v_pc, visual_features):
        """
        CHECKED!
        :param v_pc: question & v_reg pair combination      ([:, 256, 100, 100])
        :param visual_features:                             ([:, 100, 2048])
        :return: vtilde
        """

        pair_wise_rel_map = self.rp(v_pc)  # formula (5) pair-wise relationship
        inner_group_rel_map = self.rg(v_pc)  # formula (6) inner group relatioship

        __pair_wise_rel_map, vtilde = self.calculate_vtilde(inner_group_rel_map, pair_wise_rel_map, visual_features)

        # vtilde != relational_X

        return vtilde  # torch.Size([8, 100, 1024])

    @staticmethod
    def calculate_vtilde(inner_group_rel_map, pair_wise_rel_map, visual_features):
        relation_glimpse = 1
        relational_X = torch.zeros_like(visual_features)
        for g in range(relation_glimpse):
            relational_X = relational_X + torch.matmul(pair_wise_rel_map[:, g, :, :], visual_features) \
                           + torch.matmul(inner_group_rel_map[:, g, :, :], visual_features)
        relational_X = relational_X / (2 * relation_glimpse)
        return pair_wise_rel_map, relational_X

    def pair_combine(self, q, v_reg):
        """
        CHECKED
        :param q:           question features                     torch.Size([:, 1024])
        :param v_reg:           visual & geometric features concat    torch.Size([:, 100, 1024])
        :return:
        """
        num_objs = v_reg.size(1)  # 100

        # formula (2) relu(W_Q . V_reg + b_V) + relu(W_Q . Q + b_Q)
        v_reg = self.linear_v(v_reg)
        q = self.linear_q(q).unsqueeze(1)

        vtilde_reg = torch.add(v_reg, q)

        # v_pc_paper = self.calculate_v_pc(vtilde_reg)

        # formula  (3) & (4) v_pc  + v~_pc + v~_pc^T

        vtilde_pc = vtilde_reg.unsqueeze(2).expand(-1, -1, num_objs,
                                                   -1)  # repeat(1, 1, num_objs, 1)==repeat_interleave(repeats = num_objs, dim = 2)
        v_pc_mul = vtilde_pc * vtilde_pc.transpose(1, 2)

        return v_pc_mul.transpose(1, 3)

    @staticmethod
    def preprocess_features(visual_features=None, geometric_features=None, spatial_features=None):
        if utils.is_tensor_list_nonempty(visual_features) and \
                utils.is_tensor_list_nonempty(geometric_features):
            v_reg = torch.cat((visual_features, geometric_features), dim = 2)  # ([:, 100, 2052])

        if utils.is_tensor_list_nonempty(spatial_features):
            spatial_features = torch.flatten(spatial_features, start_dim = 2)
            v_reg = spatial_features
            visual_features = spatial_features

        return visual_features, v_reg

    @staticmethod
    def initialize(configs, GRID_FEAT_SIZE=None, FRCN_FEAT_SIZE=None, BBOX_FEAT_SIZE=None):
        if GRID_FEAT_SIZE:
            if len(GRID_FEAT_SIZE) == 2:
                configs.IMG_FEAT_SIZE = GRID_FEAT_SIZE[1]
                configs.VISUAL_FEATURES_DIM = GRID_FEAT_SIZE[1]
            elif len(GRID_FEAT_SIZE) == 3:
                configs.IMG_FEAT_SIZE = GRID_FEAT_SIZE[1] * GRID_FEAT_SIZE[1]
                configs.VISUAL_FEATURES_DIM = GRID_FEAT_SIZE[1] * GRID_FEAT_SIZE[1]
        elif FRCN_FEAT_SIZE and BBOX_FEAT_SIZE:
            configs.IMG_FEAT_SIZE = FRCN_FEAT_SIZE[1] + BBOX_FEAT_SIZE[1]
            configs.VISUAL_FEATURES_DIM = FRCN_FEAT_SIZE[1]

        return configs

    def forward(self, q, v_reg, visual_features):
        v_pc = self.pair_combine(q, v_reg)  # formula (2)  - (4)([:, 100,100, 256])

        vtilde = self.relational_reasoning(v_pc, visual_features)

        # from here not in author's code
        vr = vtilde.sum(dim = 1)

        vr_repr = self.vr_net(vr)
        return vr_repr
