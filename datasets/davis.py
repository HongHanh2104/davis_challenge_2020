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
                 mode=0):
        super().__init__()

        # Root directory
        assert root_path is not None, "Missing root path, should be a path DAVIS dataset!"
        self.root_path = Path(root_path)

        self.annotation = annotation_folder
        self.jpeg = jpeg_folder

        # Path to Annotations
        self.annotation_path = self.root_path / annotation_folder / resolution

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
                query_anno = video_name + "/" + pair[1]
                self.frame_list.append((support_anno, query_anno))

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))

        mode_map = [
            list(permutations(images, 2)),
            [(images[0], images[i]) for i in range(1, len(images))],
            [(i, j) for i in images for j in images if i < j]
        ]

        return mode_map[mode]

    def __getitem__(self, inx):
        support_anno_name = self.frame_list[inx][0]
        support_img_name = support_anno_name.replace("png", "jpg")
        query_anno_name = self.frame_list[inx][1]
        query_img_name = query_anno_name.replace("png", "jpg")

        print(support_anno_name, query_anno_name)

        support_img = Image.open(str(
            self.annotation_path / support_img_name).replace(self.annotation, self.jpeg)).convert('RGB')
        support_arr = np.array(support_img)
        support_img_tf = tvtf.Compose([
            tvtf.ToTensor(),
        ])
        support_img = support_img_tf(support_arr)

        query_img = Image.open(str(
            self.annotation_path / query_img_name).replace(self.annotation, self.jpeg)).convert('RGB')
        query_arr = np.array(query_img)
        query_img_tf = tvtf.Compose([
            tvtf.ToTensor(),
        ])
        query_img = query_img_tf(query_arr)

        support_anno = Image.open(
            str(self.annotation_path / support_anno_name)).convert("L")
        anno_arr = np.array(support_anno)
        anno_img_tf = tvtf.Compose([
        ])
        support_anno = torch.Tensor(anno_img_tf(anno_arr)).long()

        query_anno = Image.open(
            str(self.annotation_path / query_anno_name)).convert("L")
        anno_arr = np.array(query_anno)
        anno_img_tf = tvtf.Compose([
        ])
        query_anno = torch.Tensor(anno_img_tf(anno_arr)).long()

        return (support_img, support_anno, query_img), query_anno

    def __len__(self):
        return len(self.frame_list)
