import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
from torch.utils import data

import glob

class STM_DAVISTest(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        self.guide_dir = os.path.join(root, 'Guide', resolution) 
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = []
        N_masks = []
        N_guides = []
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            frame = torchvision.transforms.ToTensor()(Image.open(img_file).convert('RGB'))
            frame = frame.unsqueeze(0)
            N_frames.append(frame)
            try:
                guide = np.load(os.path.join(self.guide_dir, video, '{:05d}.npy'.format(f)))
                guide = torch.tensor(guide).unsqueeze(0)
                N_guides.append(guide)
            except:
                N_guides.append(torch.ones(1,*frame.shape[-2:]))

            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                mask = torch.LongTensor(np.array(Image.open(mask_file).convert('P')))
                mask = mask.unsqueeze(0)
                N_masks.append(mask)
            except:
                N_masks.append(torch.zeros(1,*frame.shape[-2:]).long())
        
        Fs = torch.cat(N_frames).transpose(0, 1)
        Ms = F.one_hot(torch.cat(N_masks), self.K).permute(3,0,1,2).float()
        Gs = torch.cat(N_guides).mean(-1)
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        
        return Fs, Ms, Gs, num_objects, info
