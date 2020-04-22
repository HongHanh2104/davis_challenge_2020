import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms as tvtf
from tqdm import tqdm

from models.canet_org import Res_Deeplab
from models.canet import CANet
from models.resnet import ResNetExtractor
from datasets.davis import DAVISDataset
from utils.device import move_to

import argparse
import matplotlib.pyplot as plt


def get_dummy(ds_size, in_channels=3, nclasses=2, h=854, w=480):
    return [
        [
            (
                torch.rand(in_channels, h, w).cpu(),
                torch.randint(low=0, high=nclasses, size=(h, w)),
                torch.rand(in_channels, h, w).cpu()
            ),
            torch.randint(low=0, high=nclasses, size=(h, w))
        ]
        for _ in range(ds_size)
    ]


def get_dataset(root, mode):
    return DAVISDataset(root_path=root, mode=mode)


if __name__ == "__main__":
    dev = torch.device('cuda')

    dataset = get_dummy(ds_size=100, h=100, w=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    extractor = ResNetExtractor('resnet50').to(dev)
    # print(extractor)
    # net = CANet(num_class=2, extractor=extractor).to(dev)
    net = Res_Deeplab(num_classes=2).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    tbar = tqdm(dataloader)
    for iter_id, (inps, lbls) in enumerate(tbar):
        inps = move_to(inps, dev)
        lbls = move_to(lbls, dev)

        outs = net(inps)

        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        tbar.set_description_str(f'{iter_id}: {loss.item()}')
