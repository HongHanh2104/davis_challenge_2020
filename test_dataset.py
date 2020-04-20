from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.davis import DAVISDataset

import argparse


def test_davis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--imgset", default="ImageSets")
    parser.add_argument("--anno", default="Annotations")
    parser.add_argument("--year", default="2017")
    parser.add_argument("--res", default="480p")
    parser.add_argument("--phase", default="train")
    parser.add_argument("--mode", default=0, type=int)
    args = parser.parse_args()

    dataset = DAVISDataset(root_path=args.root,
                           imageset_folder=args.imgset,
                           annotation_folder=args.anno,
                           year=args.year,
                           resolution=args.res,
                           phase=args.phase,
                           mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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


if __name__ == "__main__":
    test_davis()
