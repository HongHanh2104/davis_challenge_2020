import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from transforms.normalize import NormMaxMin, Normalize

from itertools import permutations
from pathlib import Path
from enum import Enum
import os
import random


class DAVISDataset(data.Dataset):

    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 mode=1,
                 max_skip=30):
        super().__init__()

        # Root directory
        assert root_path is not None, "Missing root path, should be a path DAVIS dataset!"
        self.root_path = Path(root_path)

        self.annotation = annotation_folder
        self.jpeg = jpeg_folder

        # Path to Annotations
        self.annotation_path = self.root_path / annotation_folder / resolution
        
        self.max_skip = max_skip
        # Load video name prefixes (ex: bear for bear_1)
        txt_path = self.root_path / imageset_folder / year / f"{phase}.txt"
        with open(txt_path) as files:
            video_name_prefixes = [filename.strip() for filename in files]

        # Load only the names that has prefix in video_name_prefixes
        self.video_names = []
        for folder in self.annotation_path.iterdir():
            video_id = folder.name.split('_')[0]
            if video_id in video_name_prefixes:
                self.video_names.append(folder.name)

        # Generate frames
        self.frame_list = []
        for video_name in self.video_names:
            png_pair = self.get_frame(mode, video_name)
            for pair in png_pair:
                support_anno = video_name + "/" + pair[0]
                pres_anno = video_name + "/" + pair[1]
                query_anno = video_name + "/" + pair[2]
                self.frame_list.append((support_anno, pres_anno, query_anno))

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        if mode == 0:
            return [(images[i], images[i + k - 1], images[i + k]) for k in range(2, min(n, self.max_skip)) 
                        for i in range(n - k)]
        elif mode == 1:
            return [(images[i], images[j], images[i + k]) for k in range(2, min(n, self.max_skip)) for i in range(n - k) 
                        for j in range(i + 1, i + k)]
        
    def __getitem__(self, inx):
        print(self.frame_list[inx])
        support_anno_name = self.frame_list[inx][0]
        support_img_name = support_anno_name.replace("png", "jpg")
        pres_anno_name = self.frame_list[inx][1]
        pres_img_name = pres_anno_name.replace("png", "jpg")
        query_anno_name = self.frame_list[inx][2]
        query_img_name = query_anno_name.replace("png", "jpg")

        print(support_anno_name, query_anno_name)
        
        def convert_img_tensor(img_name):
            img = Image.open(str(self.annotation_path / img_name).replace(self.annotation, self.jpeg)).convert('RGB')
            arr = np.array(img)
            img_tf = tvtf.Compose([
                tvtf.ToTensor(),
            ])
            return img_tf(arr)

        def convert_anno_tensor(anno_name):
            anno = Image.open(
                str(self.annotation_path / anno_name)).convert("L")
            arr = np.array(anno)
            anno_tf = tvtf.Compose([

            ])
            return torch.Tensor(anno_tf(arr)).long()

        support_img = convert_img_tensor(support_img_name)
        pres_img = convert_img_tensor(pres_img_name)
        query_img = convert_img_tensor(query_img_name)
        
        support_anno = convert_anno_tensor(support_anno_name)
        pres_anno = convert_anno_tensor(pres_anno_name)
        query_anno = convert_anno_tensor(query_anno_name)

        

        return (support_img, support_anno, pres_img, query_img), (pres_anno, query_anno)

    def __len__(self):
        return len(self.frame_list)
