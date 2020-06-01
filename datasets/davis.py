import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from transforms.normalize import NormMaxMin, Normalize
from transforms.crop import RandomCrop
from transforms.crop import MultiRandomCrop
from transforms.affine import MultiRandomAffine
from transforms.resize import MultiRandomResize

from itertools import permutations
from pathlib import Path
from enum import Enum
import os
import random
import json


class DAVISCoreDataset(data.Dataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 resolution="480p",
                 imageset_folder="ImageSets",
                 year="2017",
                 phase="train",
                 is_train=True,
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

        self.is_train = is_train

        # Path to Annotations
        self.annotation_path = self.root_path / annotation_folder / resolution

        # Load video name prefixes (ex: bear for bear_1)
        txt_path = self.root_path / imageset_folder / str(year) / f"{phase}.txt"
        with open(txt_path) as files:
            video_name_prefixes = [filename.strip() for filename in files]

        # Load only the names that has prefix in video_name_prefixes
        self.video_names = []
        self.infos = dict()

        print('Loading data...')
        for folder in tqdm(self.annotation_path.iterdir()):
            video_id = folder.name.split('_')[0]
            if video_id in video_name_prefixes:
                self.video_names.append(folder.name)

                self.infos[folder.name] = dict()

                # YoutubeVOS provides a meta.json file
                if os.path.exists(str(self.root_path / 'meta.json')):
                    json_data = json.load(
                        open(str(self.root_path / 'meta.json')))
                    nobjects = len(json_data['videos'][video_id]['objects'])
                    for x in folder.iterdir():
                        anno_im = Image.open(str(x)).convert('P')
                        break
                # Others might not, load all files just in case
                else:
                    nobjects = 0
                    for x in folder.iterdir():
                        anno_im = Image.open(str(x)).convert('P')
                        nobjects = max(nobjects, np.max(anno_im))
                self.infos[folder.name]['nobjects'] = nobjects
                self.infos[folder.name]['size'] = anno_im.size

                jpeg_path = self.root_path / jpeg_folder / resolution / folder.name
                self.infos[folder.name]['length'] = len(
                    list(jpeg_path.iterdir()))

    def _load_frame(self, img_name):
        # Load annotated mask
        anno_path = str(self.annotation_path / img_name)
        mask = Image.open(anno_path).convert('P')

        # Load frame image
        jpeg_path = anno_path.replace(self.annotation, self.jpeg)
        jpeg_path = jpeg_path.replace('.png', '.jpg')
        img = Image.open(jpeg_path).convert('RGB')

        # Augmentation (if train)
        if self.is_train:
            img, mask = self._augmentation(img, mask)

        # Convert to tensor
        img = tvtf.ToTensor()(img)
        mask = torch.LongTensor(np.array(mask))

        return img, mask

    def im2tensor(self, img_name):
        anno_path = str(self.annotation_path / img_name)
        jpeg_path = anno_path.replace(self.annotation, self.jpeg)
        img = Image.open(jpeg_path).convert('RGB')

        tfs = []
        if self.is_train:
            tfs.append(tvtf.Resize(384))
        tfs.append(tvtf.ToTensor())

        img_tf = tvtf.Compose(tfs)
        return img_tf(img)

    def mask2tensor(self, anno_name):
        anno_path = str(self.annotation_path / anno_name)
        anno = Image.open(anno_path).convert('P')

        tfs = []
        if self.is_train:
            tfs.append(tvtf.Resize(384))
        anno_tf = tvtf.Compose(tfs)
        ret = torch.LongTensor(np.array(anno_tf(anno)))
        ret[ret > 10] = 0
        return ret

    def _augmentation(self, img, mask):
        # img, mask = MultiRandomResize(resize_value=384)((img, mask))
        img = tvtf.Resize(384)(img)
        mask = tvtf.Resize(384, 0)(mask)
        img, mask = MultiRandomCrop(size=384)((img, mask))
        img, mask = MultiRandomAffine(degrees=(-15, 15),
                                      scale=(0.95, 1.05),
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

    def _filter(self, masks, small_obj_thres=1000):
        masks[0] = self._filter_small_objs(masks[0], small_obj_thres)
        masks = self._filter_excessive_objs(masks)
        return masks


class DAVISPairDataset(DAVISCoreDataset):
    def __init__(self,
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=-1,
                 **kwargs):
        super().__init__(**kwargs)

        self.min_skip = min_skip
        self.max_skip = max_skip
        self.max_npairs = max_npairs

        # Generate frames
        self.frame_list = []
        for video_name in self.video_names:
            nobjects = self.infos[video_name]['nobjects']
            png_pair = self.get_frame(mode, video_name)
            for pair in png_pair:
                support_anno = video_name + "/" + pair[0]
                query_anno = video_name + "/" + pair[1]
                self.frame_list.append((support_anno, query_anno, nobjects))

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
        nobjects = self.frame_list[inx][2]
        support_anno_name = self.frame_list[inx][0]
        support_img_name = support_anno_name.replace(".png", ".jpg")
        query_anno_name = self.frame_list[inx][1]
        query_img_name = query_anno_name.replace(".png", ".jpg")

        support_img = self.im2tensor(support_img_name)
        query_img = self.im2tensor(query_img_name)

        support_anno = self.mask2tensor(support_anno_name)
        query_anno = self.mask2tensor(query_anno_name)

        return (support_img, support_anno, query_img, nobjects), (query_anno,)

    def __len__(self):
        return len(self.frame_list)


class DAVISTripletDataset(DAVISCoreDataset):
    def __init__(self,
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=-1,
                 **kwargs):
        super().__init__(**kwargs)

        self.min_skip = min_skip
        self.max_skip = max_skip
        self.max_npairs = max_npairs

        # Generate frames
        self.frame_list = []
        for video_name in self.video_names:
            png_pair = self.get_frame(mode, video_name)
            nobjects = self.infos[video_name]['nobjects']
            for pair in png_pair:
                support_anno = video_name + "/" + pair[0]
                pres_anno = video_name + "/" + pair[1]
                query_anno = video_name + "/" + pair[2]
                self.frame_list.append(
                    (support_anno, pres_anno, query_anno, nobjects))

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1,
                       self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return [(images[i], images[i + k - 1], images[i + k])
                    for k in range(1 + min_skip, max_skip)
                    for i in range(n - k)]
        elif mode == 1:
            return [(images[0], images[k - 1], images[k])
                    for k in range(1 + min_skip, max_skip + 1)]
        elif mode == 2:
            indices = [(i, j, k)
                       for i in range(n-2)
                       for j in range(i+1, n-1)
                       for k in range(j+1, n)
                       if min_skip <= j - i <= max_skip and min_skip <= k - j <= max_skip]
            max_npairs = min(len(indices),
                             self.max_npairs if self.max_npairs != -1 else len(indices))
            indices = random.sample(indices, k=max_npairs)
            return [(images[i], images[j], images[k]) for i, j, k in indices]
        else:
            raise Exception('Unknown mode')

    def random_crop(self, im, mask, cropper):
        r0, c0, r1, c1 = cropper(im)
        return im[..., r0:r1, c0:c1], mask[..., r0:r1, c0:c1]

    def __getitem__(self, inx):
        support_anno_name, pres_anno_name, query_anno_name, nobjects = self.frame_list[inx]
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

        if self.is_train:
            cropper = RandomCrop(384)
            support_img, support_anno = self.random_crop(
                support_img, support_anno, cropper)
            pres_img, pres_anno = self.random_crop(
                pres_img, pres_anno, cropper)
            query_img, query_anno = self.random_crop(
                query_img, query_anno, cropper)

        return (support_img, support_anno, pres_img, query_img, nobjects), (pres_anno, query_anno)

    def __len__(self):
        return len(self.frame_list)


class DAVISPairRandomDataset(DAVISCoreDataset):
    def __init__(self,
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.mode = mode
        self.min_skip = min_skip
        self.max_skip = max_skip

        self.video_names = self.video_names * max_npairs

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1, self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return random.sample(images, 2)
        elif mode == 1:
            i = random.randrange(min_skip, max_skip + 1)
            return images[0], images[i]
        elif mode == 2:
            i = random.randrange(0, n - max_skip)
            j = i + random.randrange(min_skip, max_skip + 1)
            return images[i], images[j]
        else:
            raise Exception('Unknown mode')

    def random_crop(self, im, mask, cropper):
        r0, c0, r1, c1 = cropper(im)
        return im[..., r0:r1, c0:c1], mask[..., r0:r1, c0:c1]

    def __getitem__(self, inx):
        video_name = self.video_names[inx]
        nobjects = self.infos[video_name]['nobjects']
        support_anno_name, query_anno_name = \
            self.get_frame(self.mode, video_name)

        support_anno_name = video_name + '/' + support_anno_name
        query_anno_name = video_name + '/' + query_anno_name

        ref_img, ref_mask = self._load_frame(support_anno_name)
        query_img, query_mask = self._load_frame(query_anno_name)

        if self.is_train:
            ref_mask, query_mask = self._filter([ref_mask, query_mask])
            # if len(np.unique(ref_mask)) == 0:
            #    return self.__getitem__(inx)

        return (ref_img, ref_mask, query_img, nobjects), (query_mask,)

        '''
        support_img_name = support_anno_name.replace(".png", ".jpg")
        query_img_name = query_anno_name.replace(".png", ".jpg")

        support_img = self.im2tensor(support_img_name)
        query_img = self.im2tensor(query_img_name)

        support_anno = self.mask2tensor(support_anno_name)
        query_anno = self.mask2tensor(query_anno_name)

        if self.is_train:    
            cropper = RandomCrop(384)
            support_img, support_anno = self.random_crop(support_img, support_anno, cropper)
            query_img, query_anno = self.random_crop(query_img, query_anno, cropper)
           
            query_objs = np.unique(query_anno)
            support_objs = np.unique(support_anno)
            excess_objs = np.setdiff1d(query_objs, support_objs)
            #if len(excess_objs):
            #    return self.__getitem__(inx)
            for obj in excess_objs:
                query_anno[query_anno == obj] = 0

        return (support_img, support_anno, query_img, nobjects), (query_anno,)
        '''

    def __len__(self):
        return len(self.video_names)


class DAVISTripletRandomDataset(DAVISCoreDataset):
    def __init__(self,
                 mode=0,
                 min_skip=1,
                 max_skip=-1,
                 max_npairs=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.mode = mode
        self.min_skip = min_skip
        self.max_skip = max_skip

        self.video_names = self.video_names * max_npairs

    def get_frame(self, mode, video_name):
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        n = len(images)
        min_skip = self.min_skip
        max_skip = min(n - 1, self.max_skip if self.max_skip != -1 else n - 1)

        if mode == 0:
            return sorted(random.sample(images, 3))
        elif mode == 1:
            i = random.randrange(min_skip+1, max_skip+1)
            return images[0], images[i-1], images[i]
        elif mode == 2:
            a, b = min_skip, max_skip
            j = random.randrange(a+1, n-a)
            i = random.randrange(max(0, j - b), j - a)
            k = random.randrange(j + a, min(n, j + b))
            return images[i], images[j], images[k]
        else:
            raise Exception('Unknown mode')

    def random_crop(self, im, mask, cropper):
        r0, c0, r1, c1 = cropper(im)
        return im[..., r0:r1, c0:c1], mask[..., r0:r1, c0:c1]

    def __getitem__(self, inx):
        video_name = self.video_names[inx]
        nobjects = self.infos[video_name]['nobjects']
        support_anno_name, pres_anno_name, query_anno_name = \
            self.get_frame(self.mode, video_name)

        support_anno_name = video_name + '/' + support_anno_name
        pres_anno_name = video_name + '/' + pres_anno_name
        query_anno_name = video_name + '/' + query_anno_name

        ref_img, ref_mask = self._load_frame(support_anno_name)
        inter_img, inter_mask = self._load_frame(pres_anno_name)
        query_img, query_mask = self._load_frame(query_anno_name)

        if self.is_train:
            ref_mask, inter_mask, query_mask = self._filter(
                [ref_mask, inter_mask, query_mask])

        return (ref_img, ref_mask, inter_img, query_img, nobjects), (inter_mask, query_mask)

        '''
        support_img, pres_img, query_img = \
            map(self.im2tensor, [support_img_name,
                                 pres_img_name,
                                 query_img_name])

        support_anno, pres_anno, query_anno = \
            map(self.mask2tensor, [support_anno_name,
                                   pres_anno_name,
                                   query_anno_name])

        if self.is_train:
            cropper = RandomCrop(384)
            support_img, support_anno = self.random_crop(
                support_img, support_anno, cropper)
            pres_img, pres_anno = self.random_crop(
                pres_img, pres_anno, cropper)
            query_img, query_anno = self.random_crop(
                query_img, query_anno, cropper)

            query_objs = set(np.unique(query_anno))
            support_objs = set(np.unique(support_anno))
            pres_objs = set(np.unique(pres_anno))
            excess_objs = len(query_objs.difference(support_objs)) + \
                len(query_objs.difference(pres_objs)) + \
                len(pres_objs.difference(support_objs))
            if excess_objs:
                return self.__getitem__(inx)

        return (support_img, support_anno, pres_img, query_img, nobjects), (pres_anno, query_anno)
        '''

    def __len__(self):
        return len(self.video_names)
