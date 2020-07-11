import torch
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from itertools import permutations
from pathlib import Path
import random
import os

from transforms.crop import MultiRandomCrop
from transforms.affine import MultiRandomAffine
from transforms.resize import MultiRandomResize
from transforms.flip import MultiRandomHorizontalFlip

FSS_IMG_DIR = "JPEGImages"
FSS_ANNO_DIR = "Annotations"
FSS_IMGSETS_DIR = "ImageSets"


class FSSCoreDataset(data.Dataset):
    def __init__(self, root=None, phase=None, nrefs=1, is_train=True):
        super().__init__()

        assert root is not None, \
            'Missing root path, should be a path to an FSS directory!'

        self.root = root
        self.img_dir = os.path.join(self.root, FSS_IMG_DIR)
        self.anno_dir = os.path.join(self.root, FSS_ANNO_DIR)
        self.imgsets_dir = os.path.join(self.root, FSS_IMGSETS_DIR)

        assert isinstance(nrefs, int) and nrefs > 0, \
            'Number of references must be a positive integer!'
        self.nrefs = nrefs

        assert phase is not None, \
            f'Missing classes list, should be the name of a file in {FSS_IMGSETS_DIR}!'
        classes = open(f'{self.imgsets_dir}/{phase}.txt').readlines()
        self.classes = [x.strip() for x in classes]

        self.is_train = is_train

    def _load_frame(self, img_name, augmentation):
        # Load annotated mask
        anno_path = os.path.join(self.anno_dir, img_name)
        mask = Image.open(anno_path).convert('L')

        # Load frame image
        jpeg_path = os.path.join(self.img_dir, os.path.basename(anno_path))
        jpeg_path = jpeg_path.replace('.png', '.jpg')
        img = Image.open(jpeg_path).convert('RGB')

        # Augmentation (if train)
        if self.is_train:
            img, mask = augmentation(img, mask)

        # Convert to tensor
        img = tvtf.ToTensor()(img)
        img = tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(img)
        mask = torch.LongTensor(np.array(mask) > 0)

        return img, mask

    def _filter_small_objs(self, mask, thres):
        # Filter small objects
        ori_objs = np.unique(mask)
        for obj in ori_objs:
            area = (mask == obj).sum().item()
            if area < thres:
                mask[mask == obj] = 0
        return mask

    def _filter_excessive_objs(self, masks):
        # Filter excessive objects
        ori_objs = np.unique(masks[0])
        for i in range(1, len(masks)):
            mask_objs = np.unique(masks[i])
            excess_objs = np.setdiff1d(mask_objs, ori_objs)
            for obj in excess_objs:
                masks[i][masks[i] == obj] = 0
        return masks

    def _filter(self, masks, small_obj_thres=5000):
        masks[0] = self._filter_small_objs(masks[0], small_obj_thres)
        masks = self._filter_excessive_objs(masks)
        return masks

    def _augmentation(self, img, mask):
        img, mask = MultiRandomResize(resize_value=320)((img, mask))
        # img = tvtf.Resize((320, 320))(img)
        # mask = tvtf.Resize((320, 320), 0)(mask)
        img, mask = MultiRandomCrop(size=320)((img, mask))
        # img, mask = MultiRandomAffine(degrees=(-15, 15),
        #   scale=(0.95, 1.05),
        #   shear=(-10, 10))((img, mask))
        return img, mask


class FSSDataset(FSSCoreDataset):
    def __init__(self, max_npairs=-1, **kwargs):
        super().__init__(**kwargs)

        self.max_npairs = max_npairs

        self.tuples = []
        for _class in self.classes:
            anno_tuples = self.get_frame(_class)
            for anno_tuple in anno_tuples:
                anno_refs = [_class + "/" + x for x in anno_tuple[:-1]]
                anno_query = _class + "/" + anno_tuple[-1]
                self.tuples.append((anno_refs, anno_query))

    def get_frame(self, _class):
        images = os.listdir(os.path.join(self.anno_dir, _class))
        all_pairs = list(permutations(images, self.nrefs + 1))
        if self.max_npairs == -1:
            return all_pairs
        return random.sample(all_pairs, k=self.max_npairs)

    def __getitem__(self, inx):
        anno_ref_names, anno_query_name = self.tuples[inx]

        ref_imgs, ref_masks = [], []
        for anno_ref_name in anno_ref_names:
            ref_img, ref_mask = self._load_frame(anno_ref_name,
                                                 self._augmentation)
            ref_imgs.append(ref_img)
            ref_masks.append(ref_mask)
        query_img, query_mask = self._load_frame(anno_query_name,
                                                 self._augmentation)

        return (ref_imgs, ref_masks, query_img), query_mask

    def __len__(self):
        return len(self.tuples)


class FSSRandomDataset(FSSCoreDataset):
    count = 0

    def __init__(self, n=1, **kwargs):
        super().__init__(**kwargs)

        self.n = n
        self.classes = self.classes * self.n

        self.history = open(f'{FSSRandomDataset.count}.txt', 'w')
        FSSRandomDataset.count += 1

    def __getitem__(self, idx):
        class_id = self.classes[idx]
        path = os.path.join(self.anno_dir, class_id)

        imgs = random.sample(os.listdir(path), self.nrefs + 1)
        anno_query_name = os.path.join(class_id, imgs[-1])
        anno_ref_names = [os.path.join(class_id, x)
                          for x in imgs[:-1]]

        self.history.write(f'{anno_query_name},{",".join(anno_ref_names)}\n')

        query_img, query_mask = self._load_frame(anno_query_name,
                                                 self._augmentation)
        #if self.is_train:
        #    query_mask = self._filter_small_objs(query_mask, 5000)
        #    if query_mask.max() == 0:
        #        return self.__getitem__(idx)

        ref_imgs, ref_masks = [], []
        for anno_ref_name in anno_ref_names:
            ref_img, ref_mask = self._load_frame(anno_ref_name,
                                                 self._augmentation)

            if self.is_train:
                ref_mask, query_mask = self._filter([ref_mask, query_mask])
                if ref_mask.max() == 0:
                    return self.__getitem__(idx)
            ref_imgs.append(ref_img)
            ref_masks.append(ref_mask)

        return (ref_imgs, ref_masks, query_img), \
            (self.classes.index(class_id) + 1, (query_mask,))

    def __len__(self):
        return len(self.classes)


def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        help='path to an FSS directory')
    parser.add_argument('--phase',
                        help='name of the phase')
    parser.add_argument('--nrefs', default=1, type=int,
                        help='number of reference images')
    args = parser.parse_args()

    dataset = FSSDataset(root=args.root,
                         phase=args.phase,
                         nrefs=args.nrefs)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    for idx, batch in tqdm(enumerate(dataloader)):
        img_refs, mask_refs, img_query = batch[0]
        mask_query = batch[1]

        k = len(img_refs)
        fig, ax = plt.subplots(1, k + 1)

        for i, (img_ref, mask_ref) in enumerate(zip(img_refs, mask_refs)):
            ax[i].imshow(img_ref[0].permute(1, 2, 0))
            ax[i].imshow(mask_ref[0].squeeze(0), alpha=0.5)

        ax[k].imshow(img_query[0].permute(1, 2, 0))
        ax[k].imshow(mask_query[0].squeeze(0), alpha=0.5)

        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    test()
