import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import os
import random


class NormMaxMin():
    def __call__(self, x):
        return (x.float() - torch.min(x)) / (torch.max(x) - torch.min(x))

class DAVISLoader(data.Dataset):
    
    def __init__(self, root_path=None,
                phase="train",
                resolution="480p",
                imageset_folder=None,
                annotation_folder=None):
        super().__init__()

        assert root_path is not None, "Missing root path, should be a path DAVIS dataset!"
        assert imageset_folder is not None, "Missing imageSet folder!"
        assert annotation_folder is not None, "Missing Annotation folder!"
        
        self.root_path = Path(root_path)
        
        if phase == "train":
            txt_path = self.root_path / phase / imageset_folder / "2017" / "train.txt"
        elif phase == "val":
            txt_path = self.root_path / phase / imageset_folder / "2017" / "val.txt"
        
        # Load video name
        with open(txt_path) as files:
            self.video_names = [filename.strip() for filename in files]

        self.annotation_path = self.root_path / phase / annotation_folder / resolution 
        self.frame_list = []

        for video_name in self.video_names:
            png_pair = self.get_frame("2", video_name)

            for pair in png_pair:
                support_anno = "bear" + "/" + pair[0]
                query_anno = "bear" + "/" + pair[1]
                self.frame_list.append((support_anno, query_anno))
            
        
    def get_frame(self, mode, video_name):
        from itertools import permutations
        
        images = sorted(os.listdir(str(self.annotation_path / video_name)))
        
        mode_map = {
            "1": list(permutations(images, 2)),
            "2": [(images[0], images[i]) for i in range(1, len(images))],
            "3": [(i, j) for i in images for j in images if i < j]
        }

        return mode_map[mode]
       
            
    def __getitem__(self, inx):
        support_anno_name = self.frame_list[inx][0]
        support_img_name = support_anno_name.replace("png", "jpg")
        query_anno_name = self.frame_list[inx][1]
        query_img_name = query_anno_name.replace("png", "jpg")
        
        support_img = Image.open(str(self.annotation_path / support_img_name).replace("Annotations", "JPEGImages")).convert('RGB')
        support_arr = np.array(support_img)
        support_img_tf = tvtf.Compose([
            NormMaxMin()
        ])  #np.array(support_name)
        support_img = support_img_tf(torch.Tensor(support_arr)).unsqueeze(0)
        
        query_img = Image.open(str(self.annotation_path / query_img_name).replace("Annotations", "JPEGImages")).convert('RGB')
        query_arr = np.array(query_img)
        query_img_tf = tvtf.Compose([
            NormMaxMin()
        ])
        query_img = query_img_tf(torch.Tensor(query_arr)).unsqueeze(0)
        
        support_anno = Image.open(str(self.annotation_path / support_anno_name)).convert("L")
        anno_arr = np.array(support_anno)
        anno_img_tf = tvtf.Compose([
        ])
        support_anno = torch.Tensor(anno_img_tf(anno_arr))
        
        query_anno = Image.open(str(self.annotation_path / query_anno_name)).convert("L")
        anno_arr = np.array(query_anno)
        anno_img_tf = tvtf.Compose([
        ])
        query_anno = torch.Tensor(anno_img_tf(anno_arr))
        
        
        return [support_img, support_anno], [query_img, query_anno]

        
    def get_pair_list(self):
        pair_list = {}
        for video_name in self.video_names:
            pair_list[video_name] = [video_name + "/" + x for x in sorted(os.listdir(str(self.image_path / video_name)))]
        return pair_list 
   
def test():
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--imgset")
    parser.add_argument("--anno_folder")
    args = parser.parse_args()

    dataset = DAVISLoader(root_path=args.root, imageset_folder=args.imgset, annotation_folder=args.anno_folder)
    
    support_fr, query_fr = dataset.__getitem__(0)

    
    plt.subplot(1, 4, 1)
    plt.imshow(support_fr[0].squeeze(0))
    plt.subplot(1, 4, 2)
    plt.imshow(support_fr[1])
    plt.subplot(1, 4, 3)
    plt.imshow(query_fr[0].squeeze(0))
    plt.subplot(1, 4, 4)
    plt.imshow(query_fr[1])
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    test()
    

