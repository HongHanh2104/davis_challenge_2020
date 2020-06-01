from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.davis import *
from datasets.coco import *
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
    return parser


def visualize_pair(dataset):
    dataloader = DataLoader(dataset, batch_size=1)
    for idx, batch in enumerate(dataloader):
        for support_img, support_anno, query_img, nobjects, query_anno in zip(*batch[0], batch[1]):
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(support_img.permute(1, 2, 0))
            ax[0].imshow(support_anno.squeeze(),
                         vmin=0, vmax=nobjects, alpha=0.5)
            ax[1].imshow(query_img.permute(1, 2, 0))
            ax[1].imshow(query_anno.squeeze(),
                         vmin=0, vmax=nobjects, alpha=0.5)

            fig.tight_layout()
            plt.show()
            plt.close()


def visualize_triplet(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, batch in enumerate(dataloader):
        for a_img, a_anno, b_img, c_img, nobjects, b_anno, c_anno in zip(*batch[0], *batch[1]):
            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(a_img.permute(1, 2, 0))
            ax[0].imshow(a_anno.squeeze(),
                         vmin=0, vmax=nobjects, alpha=0.5)
            ax[1].imshow(b_img.permute(1, 2, 0))
            ax[1].imshow(b_anno.squeeze(),
                         vmin=0, vmax=nobjects, alpha=0.5)
            ax[2].imshow(c_img.permute(1, 2, 0))
            ax[2].imshow(c_anno.squeeze(),
                         vmin=0, vmax=nobjects, alpha=0.5)

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


def test_davis_random_triplet():
    parser = get_argparser()
    args = parser.parse_args()

    dataset = DAVISTripletRandomDataset(root_path=args.root,
                                        annotation_folder=args.anno,
                                        jpeg_folder=args.jpeg,
                                        resolution=args.res,
                                        imageset_folder=args.imgset,
                                        year=args.year,
                                        phase=args.phase,
                                        mode=args.mode)

    visualize_triplet(dataset)


def test_synthetic_triplet():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to the COCO dataset')
    parser.add_argument('--ann', help='The annotation subfolder (of ROOT)',
                        default='annotations')
    parser.add_argument('--img', help='The image subfolder (of ROOT)',
                        default='train2017')
    parser.add_argument('--data_type', help='train_[year] or val_[year]',
                        default='train2017')

    args = parser.parse_args()
    coco = COCODataset(root_path=args.root, data_type=args.data_type,
                       ann_folder=args.ann, img_folder=args.img)
    dataset = SyntheticTripletDataset(coco, 10)

    visualize_triplet(dataset)


if __name__ == "__main__":
    set_seed(3698)
    test_davis_random_triplet()
    # test_synthetic_triplet()
