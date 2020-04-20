####################################################################
#   From vltheanh: 
# https://github.com/vltanh/pytorch-template/blob/master/models/extractors/resnet.py

####################################################################

import torch.nn as nn
from torchvision import models

from .extractor_base import ExtractorNetwork


class ResNetExtractor(ExtractorNetwork):
    arch = {
        'resnet18': (models.resnet18(pretrained=True), 512),
        'resnet50': (models.resnet50(pretrained=True), 2048),
    }

    def __init__(self, version):
        super().__init__()
        cnn, self.feature_dim = ResNetExtractor.arch[version]
        self.extractor = nn.Sequential(*list(cnn.children())[:-4])

    def forward(self, x):
        #return self.extractor(x).view(x.size(0), -1)
        return self.extractor(x)