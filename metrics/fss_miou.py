import torch
from torch import nn
import torch.nn.functional as F


class FSSMeanIoU():
    def __init__(self, nclasses, ignore_index=None, eps=1e-9, verbose=True):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses + 1
        self.ignore_index = ignore_index
        self.eps = eps
        self.verbose = verbose

        self.reset()

    def calculate(self, output, target):
        classid, target = target
        output = output[-1]
        target = target[-1]

        nclasses = output.size(1)
        prediction = torch.argmax(output, dim=1)
        prediction = F.one_hot(prediction, nclasses).bool()
        target = F.one_hot(target, nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))
        return intersection.cpu(), union.cpu(), classid

    def update(self, value):
        for i, u, c in zip(*value):
            c = c.item()
            self.intersection[[0, c]] += i
            self.union[[0, c]] += u

    def value(self):
        ious = (self.intersection + self.eps) / (self.union + self.eps)
        miou = ious.sum()
        nclasses = ious.size(0)
        if self.ignore_index is not None:
            miou -= ious[self.ignore_index]
            nclasses -= 1
        return miou / nclasses

    def reset(self):
        self.intersection = torch.zeros(self.nclasses).float()
        self.union = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = (self.intersection + self.eps) / (self.union + self.eps)

        print(f'mIoU: {self.value():.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')
