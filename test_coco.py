from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.coco import *
import random
import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to the COCO dataset')
    parser.add_argument('--ann', help='The annotation subfolder (of ROOT)',
                        default='annotations')
    parser.add_argument('--data_type', help='train_[year] or val_[year]',
                        default='val2017')
    return parser

def test():
    parser = get_argparser()
    args = parser.parse_args()
    coco = COCODataset(root_path=args.root, data_type=args.data_type, 
                ann_folder=args.ann)

    for img, mask in coco:
        # img, mask = coco[15]
        #print(img.shape, mask.shape)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    test()
