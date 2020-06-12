import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
import random

class PASCAL5i(data.Dataset):
    def __init__(self, root_path=None,
                 annotation_folder="Annotations",
                 jpeg_folder="JPEGImages",
                 k=3):
        super().__init__()

        assert root_path is not None, "Missing root path, should be a path PASCAL-5i dataset!"
        self.root_path = Path(root_path)

        self.annotation_path = self.root_path / annotation_folder
        self.jpeg_path = self.root_path / jpeg_folder

        self.k = k
        self.classes = os.listdir(self.jpeg_path)

    def __getitem__(self, idx):
        class_id = self.classes[idx]
        path = os.path.join(self.jpeg_path, class_id)
        imgs = random.sample(os.listdir(path), self.k + 1)
        print(class_id, imgs)
        query = random.sample(imgs, 1)
        refs = [x for x in imgs if x not in query]
        
        # Query image
        img_query = Image.open(os.path.join(path, query[0])).convert('RGB')
        img_query = tvtf.ToTensor()(img_query)

        mask = Image.open(os.path.join(path.replace('JPEGImages', 'Annotations'), query[0])).convert('L')
        mask_query = torch.LongTensor(np.array(mask) > 0)

        # Ref frame
        img_refs = []
        mask_refs = []
        for ref in refs:
            img = Image.open(os.path.join(path, ref)).convert('RGB')
            img = tvtf.ToTensor()(img)
            img_refs.append(img)

            mask = Image.open(os.path.join(path.replace('JPEGImages', 'Annotations'), ref)).convert('L')
            mask = torch.LongTensor(np.array(mask) > 0)
            mask_refs.append(mask)

        return (img_refs, mask_refs, img_query), mask_query

    def __len__(self):
        return len(self.classes)

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help='path to DAVIS-like folder')
    parser.add_argument("--anno", default="Annotations",
                        help='path to Annotations subfolder (of ROOT)')
    parser.add_argument("--jpeg", default="JPEGImages",
                        help='path to JPEGImages subfolder (of ROOT)')
    parser.add_argument("--k", default=3)
    args = parser.parse_args()

    dataset = PASCAL5i(args.root, args.anno, args.jpeg, args.k)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, batch in enumerate(dataloader):
        img_refs, mask_refs, img_query = batch[0]
        mask_query = batch[1]

        k = len(img_refs)
        fig, ax = plt.subplots(1, k + 1)

        for i, (img_ref, mask_ref) in enumerate(zip(img_refs, mask_refs)):
            ax[i].imshow(img_ref[0].permute(1,2,0))
            ax[i].imshow(mask_ref[0].squeeze(0), alpha=0.5)
        
        ax[k].imshow(img_query[0].permute(1, 2, 0))
        ax[k].imshow(mask_query[0].squeeze(0), alpha=0.5)

        plt.tight_layout()
        plt.show()
        plt.close()



if __name__ == "__main__":
    test()
