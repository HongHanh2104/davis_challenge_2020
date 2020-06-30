import torch
from torchvision.transforms import functional as F

import random

class MultiRandomHorizontalFlip():
    def __init__(self, p):
        self.p = p

    def __call__(self, frames):
        if random.random() < p:
            frames = list(map(F.hflip, frames))
        return frames
