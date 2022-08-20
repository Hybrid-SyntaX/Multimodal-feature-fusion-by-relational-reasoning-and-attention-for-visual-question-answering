import torch
import torchvision.models
from torch import nn
import torch.nn.functional as F

class EnhancedCLEVRCNNFeatureExtractor(nn.Module):
    def __init__(self, configs):
        super(EnhancedCLEVRCNNFeatureExtractor, self).__init__()
        #eps= configs.EPS
        #eps = 1e-5
        self.feature_extractor = nn.Sequential(*[
            nn.Conv2d(in_channels = 3,out_channels = 128,kernel_size = (3, 3),stride = (2, 2),padding = (1, 1)),
            #nn.MaxPool2d(64), #max pool bad!
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(in_channels = 128,out_channels = 128, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
            #nn.MaxPool2d(1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
            #nn.MaxPool2d(1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
            #nn.MaxPool2d(1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
        ])
        #self.resnet=torchvision.models.resnet18(pretrained = True)

    def forward(self, x):
        #x=F.adaptive_avg_pool2d(x,(128,128))
        # import matplotlib.pyplot as plt
        # plt.imshow(x.squeeze().movedim(0, -1))
        # plt.show()
        feats = self.feature_extractor(x)
        feats = feats.transpose(1,-1)
        feats = feats.flatten(start_dim=1,end_dim=2)

        return feats # 64*128

