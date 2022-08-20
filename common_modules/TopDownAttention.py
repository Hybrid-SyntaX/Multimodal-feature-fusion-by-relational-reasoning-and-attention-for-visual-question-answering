# --------------------------------------------------------
# OpenVQA
# Written by Zhenwei Shao https://github.com/ParadoxZW
# based on the implementation in https://github.com/hengyuan-hu/bottom-up-attention-vqa
# ELU is chosen as the activation function in non-linear layers due to
# the experiment results that indicate ELU is better than ReLU in BUTD model.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


# ------------------------------
# ----- Weight Normal MLP ------
# ------------------------------


class MLP(nn.Module):
    """
    class for non-linear fully connect network
    """

    def __init__(self, dims, act='ELU', dropout_r=0.0):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout_r > 0:
                layers.append(nn.Dropout(dropout_r))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != '':
                layers.append(getattr(nn, act)())


        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x)

# ------------------------------
# ---Top Down Attention Map ----
# ------------------------------


class AttnMap(nn.Module):
    '''
    implementation of top down attention
    '''
    def __init__(self, configs):
        super(AttnMap, self).__init__()
        self.configs = configs
        self.linear_q = weight_norm(
            nn.Linear(configs.HIDDEN_SIZE, configs.HIDDEN_SIZE), dim=None)#512,512
        self.linear_v = weight_norm(
            nn.Linear(configs.IMG_FEAT_SIZE, configs.IMG_FEAT_SIZE), dim=None) #2048,2048
        self.nonlinear = MLP(
            [configs.IMG_FEAT_SIZE + configs.HIDDEN_SIZE, configs.HIDDEN_SIZE], dropout_r=configs.DROPOUT_R)
        self.linear = weight_norm(nn.Linear(configs.HIDDEN_SIZE, 1), dim=None)

    def logits(self, q, v):
        num_objs = v.size(1) #q: (8,20,1024)
        q = q.unsqueeze(1).repeat(1, num_objs, 1) #(1,100,1)

        #print(v.shape) #torch.Size([256, 100, 1024])

        vq = torch.cat((v, q), 2)



        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

    def forward(self, q, v):
        v = self.linear_v(v)
        q = self.linear_q(q)
        logits = self.logits(q, v)
        w = nn.functional.softmax(logits, 1)
        return w

# ------------------------------
# ---- Attended Joint Map ------
# ------------------------------


class TopDownAttention(nn.Module):
    def __init__(self, configs):
        super(TopDownAttention, self).__init__()

        self.configs = configs
        self.v_att = AttnMap(configs)
        self.q_net = MLP([configs.HIDDEN_SIZE, configs.HIDDEN_SIZE])
        self.v_net = MLP([configs.IMG_FEAT_SIZE, configs.HIDDEN_SIZE])

    def forward(self, q, v):
        att = self.v_att(q, v)
        atted_v = (att * v).sum(1)
        q_repr = self.q_net(q) #MLP
        v_repr = self.v_net(atted_v)
        joint_repr = q_repr * v_repr
        return joint_repr
