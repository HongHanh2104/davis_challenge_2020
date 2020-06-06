from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import SyntheticDataset
from utils.random_seed import set_seed

import random
import argparse


def visualize(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for _, batch in enumerate(dataloader):
        a_img, a_anno, *b_imgs, c_img, nobjects = batch[0]
        *b_annos, c_anno = batch[1]
        n = len(b_imgs)

        fig, ax = plt.subplots(1, n + 2)
        ax[0].imshow(a_img[0].permute(1, 2, 0))
        ax[0].imshow(a_anno[0].squeeze(),
                     vmin=0, vmax=nobjects, alpha=0.5)

        for i, (b_img, b_anno) in enumerate(zip(b_imgs, b_annos)):
            ax[1+i].imshow(b_img[0].permute(1, 2, 0))
            ax[1+i].imshow(b_anno[0].squeeze(),
                           vmin=0, vmax=nobjects, alpha=0.5)

        ax[n+1].imshow(c_img[0].permute(1, 2, 0))
        ax[n+1].imshow(c_anno[0].squeeze(),
                       vmin=0, vmax=nobjects, alpha=0.5)

        fig.tight_layout()
        plt.show()
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to the COCO dataset')
    parser.add_argument('--ann', help='The annotation subfolder (of ROOT)',
                        default='annotations')
    parser.add_argument('--img', help='The image subfolder (of ROOT)',
                        default='train2017')
    parser.add_argument('--data_type', help='train_[year] or val_[year]',
                        default='train2017')
    parser.add_argument('--k', help='number of generated images (including the first)',
                        default=3, type=int)
    parser.add_argument('--n', help='number of iterations',
                        default=100, type=int)
    return parser.parse_args()


def test_synthetic(args):
    dataset = SyntheticDataset({
        'name': 'COCODataset',
        'args': {
            'root_path': args.root,
            'data_type': args.data_type,
            'ann_folder': args.ann,
            'img_folder': args.img,
        },
    }, niters=args.n, nimgs=args.k)

    visualize(dataset)


if __name__ == "__main__":
    set_seed(3698)
    args = parse_args()
    test_synthetic(args)
