import torch
from torch import nn
import torch.nn.functional as F

from utils.segmentation import multi_class_prediction, binary_prediction


class MeanIoU():
    def __init__(self, nclasses, ignore_index=None, eps=1e-6):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def calculate(self, output, target):
        nclasses = output.size(1)
        prediction = torch.argmax(output, dim=1)
        prediction = F.one_hot(prediction, nclasses).bool()
        target = F.one_hot(target, nclasses).bool()
        intersection = (prediction & target).sum((-3, -2))
        union = (prediction | target).sum((-3, -2))
        ious = (intersection.float() + self.eps) / (union.float() + self.eps)
        return ious.cpu()

    def update(self, value):
        self.mean_class += value.sum(0)
        self.sample_size += value.size(0)

    def value(self):
        ious = self.mean_class
        miou = ious.sum() / self.sample_size
        nclasses = ious.size(0)
        if self.ignore_index is not None:
            miou -= ious[self.ignore_index] / self.sample_size
            nclasses -= 1
        return miou / nclasses

    def reset(self):
        self.mean_class = torch.zeros(self.nclasses).float()
        self.sample_size = 0

    def summary(self):
        class_iou = self.mean_class / self.sample_size

        print(f'mIoU: {self.value():.6f}')
        for i, x in enumerate(class_iou):
            print(f'\tClass {i:3d}: {x:.6f}')


class ModifiedMeanIoU(MeanIoU):
    def calculate(self, output, target):
        return super().calculate(output[-1], target[-1])
