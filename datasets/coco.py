import torch
import torchvision.transforms as tvtf
from torch.utils import data
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import requests
import io
import random

from transforms.crop import MultiRandomCrop
from transforms.affine import MultiRandomAffine
from transforms.resize import MultiRandomResize

from utils import getter

class COCODataset(data.Dataset):
    def __init__(self, 
                root_path=None,
                data_type='train2017',
                ann_folder='annotations',
                img_folder='train2017'):
        super().__init__()

        assert root_path is not None, "Missing Missing root path, should be a path COCO dataset!"

        self.root_path = Path(root_path)
        
        file_json = 'instances_{}.json'.format(data_type)
        annfiles_path = self.root_path / ann_folder / file_json

        self.img_folder = self.root_path / img_folder

        # initialize COCO api for instance annotations
        self.coco = COCO(annfiles_path)

        # Get all images
        self.imgIds = sorted(self.coco.getImgIds())
    
    def _get_mask(self, h, w, anns):
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            combined_mask = np.maximum(combined_mask, (i + 1) * mask)
        return combined_mask
    
    def __getitem__(self, inx):
        imgId = self.imgIds[inx]
        imgId = self.coco.loadImgs(imgId)[0]

        #response = requests.get(imgId['coco_url'])
        #img_bytes = io.BytesIO(response.content)
        path = self.img_folder / imgId['file_name']
        img = Image.open(path).convert('RGB')
        # Get all annotations of this image
        annIds = self.coco.getAnnIds(imgIds=imgId['id'], iscrowd=False)
        
        # Choose randomly 3 in annIds
        annIds = random.sample(annIds, k=min(3, len(annIds)))
        
        anns = self.coco.loadAnns(annIds)
        mask = self._get_mask(imgId['height'], imgId['width'], anns)
        mask = Image.fromarray(mask).convert('P')

        return img, mask

    def __len__(self):
        return len(self.imgIds)

class SyntheticDataset:
    def __init__(self, dataset, niters):
        self.dataset = getter.get_instance(dataset)
        self.niters = niters
        self.k = 3

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
                           for _ in range(self.k)])
        im_0, *ims = map(tvtf.ToTensor(), ims)
        mask2tensor = lambda x: torch.LongTensor(np.array(x))
        masks = list(map(mask2tensor, masks))
        mask_0, *masks = self._filter(masks)
        nobjects = mask_0.max()
        if nobjects == 0:
            return self.__getitem__(i)
        return (im_0, mask_0, *ims, nobjects), (*masks)
    
    def __len__(self):
        return self.niters
