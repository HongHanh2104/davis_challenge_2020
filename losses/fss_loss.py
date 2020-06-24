import torch
from torch import nn

from .classification.crossentropy import CrossEntropyLoss


class FSSCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = CrossEntropyLoss(**kwargs)

    def forward(self, output, target):
        _, target = target
        loss = 0.0
        for pred, true in zip(output, target):
            loss += self.loss(pred, true)
        return loss
