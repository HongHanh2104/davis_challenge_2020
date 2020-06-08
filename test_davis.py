from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import *
from utils.random_seed import set_seed

import argparse


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help='path to DAVIS-like folder')
    parser.add_argument("--anno", default="Annotations",
                        help='path to Annotations subfolder (of ROOT)')
    parser.add_argument("--jpeg", default="JPEGImages",
                        help='path to JPEGImages subfolder (of ROOT)')
    parser.add_argument("--res", default="480p",
                        help='path to Resolution subfolder (of ANNO and JPEG)')
    parser.add_argument("--imgset", default="ImageSets",
                        help='path to ImageSet subfolder (of ROOT)')
    parser.add_argument("--year", default="2017",
                        help='path to Year subfolder (of IMGSET)')
    parser.add_argument("--phase", default="train",
                        help='path to phase txt file (of IMGSET/YEAR)')
    parser.add_argument("--mode", default=2, type=int,
                        help='frame pair selector mode')
    parser.add_argument("--train", action='store_true',
                        help='load as train dataset')
    parser.add_argument("--max_skip", default=-1, type=int,
                        help='maximum number of gap between frames')
    parser.add_argument("--min_skip", default=1, type=int,
                        help='minimum number of gap between frames')
    return parser


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


def test_davis_pair(parser):
    args = parser.parse_args()

    return DAVISPairDataset(root_path=args.root,
                            annotation_folder=args.anno,
                            jpeg_folder=args.jpeg,
                            resolution=args.res,
                            imageset_folder=args.imgset,
                            year=args.year,
                            phase=args.phase,
                            mode=args.mode,
                            is_train=args.train)


def test_davis_random_pair(parser):
    args = parser.parse_args()

    return DAVISPairRandomDataset(root_path=args.root,
                                  annotation_folder=args.anno,
                                  jpeg_folder=args.jpeg,
                                  resolution=args.res,
                                  imageset_folder=args.imgset,
                                  year=args.year,
                                  phase=args.phase,
                                  mode=args.mode,
                                  is_train=args.train)


def test_davis_triplet(parser):
    args = parser.parse_args()

    return DAVISTripletDataset(root_path=args.root,
                               annotation_folder=args.anno,
                               jpeg_folder=args.jpeg,
                               resolution=args.res,
                               imageset_folder=args.imgset,
                               year=args.year,
                               phase=args.phase,
                               mode=args.mode,
                               is_train=args.train)


def test_davis_random_triplet(parser):
    args = parser.parse_args()

    return DAVISTripletRandomDataset(root_path=args.root,
                                     annotation_folder=args.anno,
                                     jpeg_folder=args.jpeg,
                                     resolution=args.res,
                                     imageset_folder=args.imgset,
                                     year=args.year,
                                     phase=args.phase,
                                     mode=args.mode,
                                     is_train=args.train)


if __name__ == "__main__":
    set_seed(3698)

    parser = get_argparser()
    dataset = test_davis_triplet(parser)
    visualize(dataset)
