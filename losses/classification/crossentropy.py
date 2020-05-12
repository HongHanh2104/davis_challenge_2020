import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return self.loss(output, target)


class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, beta, **kwargs):
        if isinstance(beta, (float, int)):
            self.beta = torch.Tensor([beta])
        if isinstance(beta, list):
            self.beta = torch.Tensor(beta)
        super().__init__(pos_weight=self.beta, **kwargs)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)


class MultiCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = CrossEntropyLoss(**kwargs)

    def forward(self, output, target):
        w = 1.0 / len(output)
        loss = 0.0
        for pred, true in zip(output, target):
            loss += self.loss(pred, true) * w
        return loss
