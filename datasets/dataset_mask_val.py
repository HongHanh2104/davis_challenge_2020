import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time

class FSS_ValDataset(object):

    def __init__(self, data_dir, fold, 
                 input_size=[321, 321], 
                 normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1]):
        self.data_dir = data_dir
        self.input_size = input_size
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)
        self.chosen_data_list=self.chosen_data_list_1+chosen_data_list_2+chosen_data_list_3
        self.chosen_data_list=self.chosen_data_list[:1000]

        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.query_class_support_list=[None] * 1000
        for index in range (1000):
            query_name=self.chosen_data_list[index][0]
            sample_class=self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
            while True:  # random sample a support data
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    break
            self.query_class_support_list[index]=[query_name,sample_class,support_name]
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.fold = fold

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        f = open(os.path.join(self.data_dir, 'Binary_map_aug','val', 'split%1d_val.txt' % (fold)))
        while True:
            item = f.readline()
            if item == '':
                break
            img_name = item[:11]
            cat = int(item[13:15])
            new_exist_class_list.append([img_name, cat])
        return new_exist_class_list


    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map_aug', 'val', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]
        support_name=self.query_class_support_list[index][2]
        support_rgb = self.normalize(self.ToTensor(Image.open(os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))
        support_mask = self.ToTensor(Image.open(os.path.join(self.data_dir, 'Binary_map_aug', 'val', str(sample_class), support_name + '.png')))
        query_rgb = self.normalize(self.ToTensor(Image.open(os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))
        query_mask = self.ToTensor(Image.open(os.path.join(self.data_dir, 'Binary_map_aug', 'val', str(sample_class),query_name + '.png')))
        return ([support_rgb], [support_mask[0].long()], query_rgb), (sample_class - 5 * self.fold, (support_mask[0].long(), query_mask[0].long()))

    def __len__(self):
        return 1000
