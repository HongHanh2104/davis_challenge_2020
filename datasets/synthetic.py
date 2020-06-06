import torch
from torchvision import transforms as tvtf
import numpy as np

from transforms.crop import MultiRandomCrop
from transforms.affine import MultiRandomAffine
from transforms.resize import MultiRandomResize
from utils import getter

import random


class SyntheticDataset:
    def __init__(self, dataset, niters, nimgs):
        self.dataset = getter.get_instance(dataset)
        self.niters = niters
        self.nimgs = nimgs

    def _augmentation(self, img, mask):
        img, mask = MultiRandomResize(resize_value=384)((img, mask))
        #img = tvtf.Resize(384)(img)
        #mask = tvtf.Resize(384, 0)(mask)
        img, mask = MultiRandomCrop(size=384)((img, mask))
        img, mask = MultiRandomAffine(degrees=(-20, 20),
                                      scale=(0.9, 1.1),
                                      shear=(-10, 10))((img, mask))
        return img, mask

    def _filter_small_objs(self, mask, thres):
        # Filter small objects
        ori_objs = np.unique(mask)
        for obj in ori_objs:
            area = (mask == obj).sum().item()
            if area < thres:
                mask[mask == obj] = 0
        return mask

    def _filter_excessive_objs(self, masks):
        # Filter excessive objects
        ori_objs = np.unique(masks[0])
        for i in range(1, len(masks)):
            mask_objs = np.unique(masks[i])
            excess_objs = np.setdiff1d(mask_objs, ori_objs)
            for obj in excess_objs:
                masks[i][masks[i] == obj] = 0
        return masks

    def _filter(self, masks):
        masks[0] = self._filter_small_objs(masks[0], 1000)
        masks = self._filter_excessive_objs(masks)
        return masks

    def __getitem__(self, i):
        i = random.randrange(0, len(self.dataset))
        ori_img, ori_mask = self.dataset[i]
        ims, masks = zip(*[self._augmentation(ori_img, ori_mask)
                           for _ in range(self.nimgs)])
        im_0, *ims = map(tvtf.ToTensor(), ims)
        def mask2tensor(x): return torch.LongTensor(np.array(x))
        masks = list(map(mask2tensor, masks))
        mask_0, *masks = self._filter(masks)
        nobjects = mask_0.max()
        if nobjects == 0:
            return self.__getitem__(i)
        return (im_0, mask_0, *ims, nobjects), tuple(masks)

    def __len__(self):
        return self.niters
