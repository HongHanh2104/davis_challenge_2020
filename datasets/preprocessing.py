import argparse
from pathlib import Path
import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Target folder structure

FSS_IMG_DIR = "JPEGImages"
FSS_ANNO_DIR = "Annotations"
FSS_IMGSETS_DIR = "ImageSets"


def kfold_split(classes, random_split=False, k=5):
    if random_split:
        classes = random.shuffle(classes)

    folds = []
    for i in range(k):
        val_set = classes[k*i:k*(i+1)]
        train_set = classes[:k*i] + classes[k*(i+1):]
        folds.append((train_set, val_set))
    return folds


def create_target_directory(root, classes):
    '''
        Target directory structure:
        - root
            - ImageSets
            - JPEGImages
                - aeroplane
                    - 2007_000027.jpg
                    - ...
                - ...
            - Annotations
                - aeroplane
                    - 2007_000027.jpg
                - ...
    '''

    # Create root
    os.makedirs(root, exist_ok=True)

    # Create ImageSets folder
    os.makedirs(f"{root}/{FSS_IMGSETS_DIR}", exist_ok=True)

    # Create JPEGImages and Annotations
    for x in [FSS_IMG_DIR, FSS_ANNO_DIR]:
        for y in classes:
            os.makedirs(f"{root}/{x}/{y}", exist_ok=True)

# ========================= PASCAL_5i =========================


PASCAL_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

PASCAL_IMG_DIR = 'JPEGImages'
PASCAL_ANNO_DIR = 'SegmentationClass'


def create_class_txt(root, classes, dst_folder):
    namefile = "classes.txt"
    with open(root / dst_folder / "ImageSets" / namefile, "w") as f:
        for c in classes:
            f.write(c + "\n")


def pascal2fss(data_folder, dst_folder):
    # Create directory
    create_target_directory(dst_folder, PASCAL_CLASS)

    data_folder = Path(data_folder)
    root_img_path = data_folder / PASCAL_IMG_DIR
    segclass_path = data_folder / PASCAL_ANNO_DIR

    dst_folder = Path(dst_folder)
    dst_img_path = dst_folder / FSS_IMG_DIR
    dst_mask_path = dst_folder / FSS_ANNO_DIR

    files = os.listdir(segclass_path)
    progress_bar = tqdm(files)
    for _file in progress_bar:
        progress_bar.set_description_str(_file)

        # Load mask annotation
        path = segclass_path / _file
        mask = Image.open(path).convert('P')

        # Get all unique object ids
        class_id = np.unique(mask)
        class_id = [x for x in class_id if 0 < x < 255]

        # Copy JPEGImages into class folders
        for x in class_id:
            os.system('cp "{}/{}" "{}/{}"'.format(root_img_path,
                                                  _file.replace('png', 'jpg'),
                                                  dst_img_path,
                                                  PASCAL_CLASS[x - 1]))

        # Copy individual masks
        mask = np.array(mask)
        for _id in class_id:
            copy_mask = (mask == _id).astype(np.uint8)
            save_mask = Image.fromarray(copy_mask, 'P')
            save_mask.putpalette([0, 0, 0,
                                  255, 255, 255])

            save_mask.save('{}/{}/{}'.format(dst_mask_path,
                                             PASCAL_CLASS[_id - 1],
                                             _file))


def gen_pascal_5i(fss_folder):
    imgsets_dir = fss_folder + '/' + FSS_IMGSETS_DIR
    folds = kfold_split(PASCAL_CLASS)
    for i, (train, val) in enumerate(folds):
        open(f'{imgsets_dir}/{i}_train.txt', 'w').write('\n'.join(train))
        open(f'{imgsets_dir}/{i}_val.txt', 'w').write('\n'.join(val))


def visualize(root):
    import random

    root = Path(root)
    img_path = root / IMAGE_FOLDER
    mask_path = root / MASK_FOLDER
    classes = random.sample(PASCAL_CLASS, 5 + 1)
    for c in classes:
        imgs = random.sample(os.listdir(img_path / c), 3 + 1)
        print(c)
        for i in range(len(imgs)):
            img = Image.open(img_path / c / imgs[i]).convert('RGB')
            mask = Image.open(
                mask_path / c / imgs[i].replace('jpg', 'png')).convert('L')
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[1].imshow(mask)
            plt.tight_layout()
            plt.show()
            plt.close()


def fss1000_preprocess(root,
                       data_folder,
                       img_folder='JPEGImages',
                       mask_folder='Annotations',
                       dst_folder='new_fss1000'):
    root_path = Path(root)

    classes = os.listdir(root_path / data_folder)
    print(type(classes[0]), type(PASCAL_CLASS[0]))
    create_data_structure(root, classes, dst_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to dataset')
    parser.add_argument('--data_folder', help='The folder of dataset')
    parser.add_argument('--dst_folder', help='The destination folder.')
    args = parser.parse_args()

    pascal2fss(args.data_folder, args.dst_folder)
    gen_pascal_5i(args.dst_folder)

    # visualize(os.path.join(args.root, args.dst_folder))

    #fss1000_preprocess(args.root, args.data_folder, args.img_folder, args.mask_folder, args.dst_folder)
