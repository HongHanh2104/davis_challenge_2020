from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.davis import *

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
    return parser


def visualize_pair(dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    for idx, batch in enumerate(dataloader):
        for support_img, support_anno, query_img, query_anno in zip(*batch[0], batch[1]):
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(support_img.permute(1, 2, 0))
            ax[0, 1].imshow(support_anno.squeeze())
            ax[1, 0].imshow(query_img.permute(1, 2, 0))
            ax[1, 1].imshow(query_anno.squeeze())

            fig.tight_layout()
            plt.show()
            plt.close()


def visualize_triplet(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, batch in enumerate(dataloader):
        for a_img, a_anno, b_img, c_img, b_anno, c_anno in zip(*batch[0], *batch[1]):
            fig, ax = plt.subplots(3, 2)
            ax[0, 0].imshow(a_img.permute(1, 2, 0))
            ax[0, 1].imshow(a_anno.squeeze())
            ax[1, 0].imshow(b_img.permute(1, 2, 0))
            ax[1, 1].imshow(b_anno.squeeze())
            ax[2, 0].imshow(c_img.permute(1, 2, 0))
            ax[2, 1].imshow(c_anno.squeeze())

            fig.tight_layout()
            plt.show()
            plt.close()


def test_davis_pair():
    parser = get_argparser()
    args = parser.parse_args()

    dataset = DAVISPairDataset(root_path=args.root,
                               annotation_folder=args.anno,
                               jpeg_folder=args.jpeg,
                               resolution=args.res,
                               imageset_folder=args.imgset,
                               year=args.year,
                               phase=args.phase,
                               mode=args.mode)

    visualize_pair(dataset)


def test_davis_random_pair():
    parser = get_argparser()
    args = parser.parse_args()

    dataset = DAVISPairRandomDataset(root_path=args.root,
                                     annotation_folder=args.anno,
                                     jpeg_folder=args.jpeg,
                                     resolution=args.res,
                                     imageset_folder=args.imgset,
                                     year=args.year,
                                     phase=args.phase,
                                     mode=args.mode)

    visualize_pair(dataset)


def test_davis_triplet():
    parser = get_argparser()
    args = parser.parse_args()

    dataset = DAVISTripletDataset(root_path=args.root,
                                  annotation_folder=args.anno,
                                  jpeg_folder=args.jpeg,
                                  resolution=args.res,
                                  imageset_folder=args.imgset,
                                  year=args.year,
                                  phase=args.phase,
                                  mode=args.mode)

    visualize_triplet(dataset)


if __name__ == "__main__":
    test_davis_triplet()
