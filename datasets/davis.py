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


class DAVISCoreDataset(data.Dataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=-1):
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

    def im2tensor(self, img_name):
        anno_path = str(self.annotation_path / img_name)
        jpeg_path = anno_path.replace(self.annotation, self.jpeg)
        img = Image.open(jpeg_path).convert('RGB')
        img_tf = tvtf.Compose([
            tvtf.ToTensor(),
        ])
        return img_tf(img)

    def mask2tensor(self, anno_name):
        anno_path = str(self.annotation_path / anno_name)
        anno = Image.open(anno_path).convert('P')
        anno_tf = tvtf.Compose([
        ])
        return torch.LongTensor(np.array(anno_tf(anno)))


class DAVISPairDataset(DAVISCoreDataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=-1):
        super().__init__(root_path, annotation_folder, jpeg_folder,
                         resolution, imageset_folder, year, phase)

        self.min_skip = min_skip
        self.max_skip = max_skip
        self.max_npairs = max_npairs

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
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1, self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return list(permutations(images, 2))
        elif mode == 1:
            return [(images[0], images[i]) for i in range(1, n) if max_skip >= i >= min_skip]
        elif mode == 2:
            indices = [(i, j) for i in range(n-1)
                       for j in range(i+1, n) if max_skip >= j - i >= min_skip]
            max_npairs = min(len(indices),
                             self.max_npairs if self.max_npairs != -1 else len(indices))
            indices = random.sample(indices, k=max_npairs)
            return [(images[i], images[j]) for i, j in indices]
        else:
            raise Exception('Unknown mode')

    def __getitem__(self, inx):
        support_anno_name = self.frame_list[inx][0]
        support_img_name = support_anno_name.replace(".png", ".jpg")
        query_anno_name = self.frame_list[inx][1]
        query_img_name = query_anno_name.replace(".png", ".jpg")

        support_img = self.im2tensor(support_img_name)
        query_img = self.im2tensor(query_img_name)

        support_anno = self.mask2tensor(support_anno_name)
        query_anno = self.mask2tensor(query_anno_name)

        return (support_img, support_anno, query_img), query_anno

    def __len__(self):
        return len(self.frame_list)


class DAVISTripletDataset(DAVISCoreDataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=-1):
        super().__init__(root_path, annotation_folder, jpeg_folder,
                         resolution, imageset_folder, year, phase)

        self.min_skip = min_skip
        self.max_skip = max_skip
        self.max_npairs = max_npairs

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
        min_skip = self.min_skip
        max_skip = min(n // 2 - (n + 1) % 2,
                       self.max_skip if self.max_skip != -1 else n // 2 - (n + 1) % 2)

        if mode == 0:
            return [(images[i], images[i + k - 1], images[i + k])
                    for k in range(1 + min_skip, max_skip)
                    for i in range(n - k)]
        elif mode == 1:
            return [(images[0], images[k - 1], images[k])
                    for k in range(1 + min_skip, max_skip)]
        elif mode == 2:
            indices = [(i, j, k)
                       for i in range(n-2*max_skip)
                       for j in range(i+min_skip, i+max_skip+1)
                       for k in range(j+min_skip, j+max_skip+1)]
            max_npairs = min(len(indices),
                             self.max_npairs if self.max_npairs != -1 else len(indices))
            indices = random.sample(indices, k=max_npairs)
            return [(images[i], images[j], images[k]) for i, j, k in indices]
        else:
            raise Exception('Unknown mode')

    def __getitem__(self, inx):
        support_anno_name, pres_anno_name, query_anno_name = self.frame_list[inx]
        support_img_name, pres_img_name, query_img_name = \
            map(lambda x: x.replace('png', 'jpg'), [support_anno_name,
                                                    pres_anno_name,
                                                    query_anno_name])

        support_img, pres_img, query_img = \
            map(self.im2tensor, [support_img_name,
                                 pres_img_name,
                                 query_img_name])

        support_anno, pres_anno, query_anno = \
            map(self.mask2tensor, [support_anno_name,
                                   pres_anno_name,
                                   query_anno_name])

        return (support_img, support_anno, pres_img, query_img), (pres_anno, query_anno)

    def __len__(self):
        return len(self.frame_list)


class DAVISPairRandomDataset(DAVISCoreDataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=-1):
        super().__init__(root_path, annotation_folder, jpeg_folder,
                         resolution, imageset_folder, year, phase)

        self.mode = mode
        self.min_skip = min_skip
        self.max_skip = max_skip

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1, self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return random.sample(images, 2)
        elif mode == 1:
            i = random.randint(min_skip, max_skip + 1)
            return images[0], images[i]
        elif mode == 2:
            i = random.randint(0, n - max_skip)
            j = i + random.randint(min_skip, max_skip + 1)
            return images[i], images[j]
        else:
            raise Exception('Unknown mode')

    def __getitem__(self, inx):
        video_name = self.video_names[inx]
        support_anno_name, query_anno_name = \
            self.get_frame(self.mode, video_name)
        support_anno_name = video_name + '/' + support_anno_name
        query_anno_name = video_name + '/' + query_anno_name

        support_img_name = support_anno_name.replace(".png", ".jpg")
        query_img_name = query_anno_name.replace(".png", ".jpg")

        support_img = self.im2tensor(support_img_name)
        query_img = self.im2tensor(query_img_name)

        support_anno = self.mask2tensor(support_anno_name)
        query_anno = self.mask2tensor(query_anno_name)

        return (support_img, support_anno, query_img), query_anno

    def __len__(self):
        return len(self.video_names)
