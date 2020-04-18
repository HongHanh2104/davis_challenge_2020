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
                image_folder=None):
        super().__init__()

        assert root_path is not None, "Missing root path, should be a path DAVIS dataset!"
        assert imageset_folder is not None, "Missing imageSet folder!"
        assert image_folder is not None, "Missing JPEGImages folder!"
        
        self.root_path = Path(root_path)
        
        if phase == "train":
            txt_path = self.root_path / phase / imageset_folder / "2017" / "train.txt"
        elif phase == "val":
            txt_path = self.root_path / phase / imageset_folder / "2017" / "val.txt"
        
        # Load video name
        with open(txt_path) as files:
            self.video_names = [filename.strip() for filename in files]
            
        self.image_path = self.root_path / phase / image_folder / resolution 
        self.frame_list = []
        for video_name in self.video_names:
            for filename in sorted(os.listdir(str(self.image_path / video_name))):
                img = video_name + "/" + filename
                ann = img.replace("jpg", "png")
                self.frame_list.append([img, ann])
            
    def __getitem__(self, inx):
        support_name = self.frame_list[inx][0]
        anno_name = self.frame_list[inx][1]

        pair_list = self.get_pair_list()
        query_img_list = pair_list[anno_name.split('/')[0]]
        
        while True:
            query_name = query_img_list[random.randint(0, len(query_img_list) - 1)]
            if query_name != support_name:
                break
        
        support_img = Image.open(str(self.image_path / support_name)).convert('RGB')
        support_arr = np.array(support_img)
        support_img_tf = tvtf.Compose([
            NormMaxMin()
        ])  #np.array(support_name)
        support_img = support_img_tf(torch.Tensor(support_arr)).unsqueeze(0)
        
        query_img = Image.open(str(self.image_path / query_name)).convert('RGB')
        query_arr = np.array(query_img)
        query_img_tf = tvtf.Compose([
            NormMaxMin()
        ])
        query_img = query_img_tf(torch.Tensor(query_arr)).unsqueeze(0)
        
        anno_img = Image.open(str(self.image_path / anno_name).replace("JPEGImages", "Annotations")).convert("L")
        anno_arr = np.array(anno_img)
        anno_img_tf = tvtf.Compose([
        ])
        anno_img = torch.Tensor(anno_img_tf(anno_arr))
        print(support_img.shape, anno_img.shape, query_img.shape)
        return [support_img, anno_img], query_img

        
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
    parser.add_argument("--img_folder")
    args = parser.parse_args()

    dataset = DAVISLoader(root_path=args.root, imageset_folder=args.imgset, image_folder=args.img_folder)
    #dataset.__getitem__(0)
    support_fr, query_img = dataset.__getitem__(0)

    
    plt.subplot(1, 3, 1)
    plt.imshow(support_fr[0].squeeze(0))
    plt.subplot(1, 3, 2)
    plt.imshow(support_fr[1])
    plt.subplot(1, 3, 3)
    plt.imshow(query_img.squeeze(0))
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    test()
    

