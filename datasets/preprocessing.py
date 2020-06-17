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

FSS_IMG_DIR = 'JPEGImages'
FSS_ANNO_DIR = 'Annotations'
FSS_IMGSETS_DIR = 'ImageSets'


def kfold_split(classes, random_split=False, k=5):
    if random_split:
        classes = random.shuffle(classes)

    folds = []
    for i in range(k):
        val_set = classes[k*i:k*(i+1)]
        train_set = classes[:k*i] + classes[k*(i+1):]
        folds.append((train_set, val_set))
    return folds


def create_target_directory(dst_folder, classes):
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
    os.makedirs(dst_folder, exist_ok=True)

    # Create ImageSets folder
    os.makedirs(f'{dst_folder}/{FSS_IMGSETS_DIR}', exist_ok=True)

    # Create JPEGImages and Annotations
    os.makedirs(f'{dst_folder}/{FSS_IMG_DIR}', exist_ok=True)    
    for y in classes:
        os.makedirs(f'{dst_folder}/{FSS_ANNO_DIR}/{y}', exist_ok=True)

# ========================= PASCAL_5i =========================


PASCAL_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

PASCAL_IMG_DIR = 'JPEGImages'
PASCAL_ANNO_DIR = 'SegmentationClass'
PASCAL_IMAGESET_DIR = 'ImageSets'


def create_class_txt(root, classes, dst_folder):
    namefile = 'classes.txt'
    with open(root / dst_folder / PASCAL_IMAGESET_DIR / namefile, 'w') as f:
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
            os.system('cp "{}/{}" "{}"'.format(root_img_path,
                                                  _file.replace('png', 'jpg'),
                                                  dst_img_path))

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

# ========================= FSS-1000 =========================
FSS1000_DATA = "fewshot_data"

def fss1000tofss(data_folder,
                 dst_folder):
    data_path = Path(data_folder)
    classes = os.listdir(data_path / FSS1000_DATA)
    # Create target directory 
    create_target_directory(dst_folder, classes)

    dst_path = Path(dst_folder)
    dst_img_path = dst_path / FSS_IMG_DIR
    dst_mask_path = dst_path / FSS_ANNO_DIR

    files = os.listdir(data_path / FSS1000_DATA)
    
    progress_bar = tqdm(files)
    for _file in progress_bar:
        progress_bar.set_description_str(_file)
        imgs = glob.glob(str(data_path / FSS1000_DATA / _file / '*.jpg'))
        [os.system(f'cp "{x}" "{dst_img_path}/{_file}"') for x in imgs]
        
        masks = [x.replace('jpg', 'png') for x in imgs]
        [os.system(f'cp "{x}" "{dst_mask_path}/{_file}"') for x in masks]

############################### Celeb ####################################
CELEB_CLASS = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', \
               'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', \
               'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', \
               'neck_l', 'neck', 'cloth']

CELEB_IMG_DIR = 'CelebA-HQ-img'
CELEB_MASK_DIR = 'CelebAMask-HQ-mask-anno'

def celeb2fss(data_folder, dst_folder):
    data_path = Path(data_folder)
    data_img_path = data_path / CELEB_IMG_DIR
    data_mask_path = data_path / CELEB_MASK_DIR 

    dst_path = Path(dst_folder)
    dst_img_path = dst_path / FSS_IMG_DIR
    dst_mask_path = dst_path / FSS_ANNO_DIR

    
    create_target_directory(dst_folder, CELEB_CLASS)

    img_files = os.listdir(data_img_path)
    img_progress_bar = tqdm(img_files)
    for _file in img_progress_bar:
        img_progress_bar.set_description_str(_file)
        [os.system(f'cp "{data_img_path}/{_file}" "{dst_img_path}/"')]
    print("Done copy JPEGImages folder.")
    
    subfolders = os.listdir(data_mask_path)
    mask_progress_bar = tqdm(subfolders)
    for folder in mask_progress_bar:
        mask_progress_bar.set_description_str(folder)
        #[print(x[6:-4]) for x in os.listdir(data_mask_path / folder)]
        [os.system(f'cp "{data_mask_path}/{folder}/{x}" "{dst_mask_path}/{x[6:-4]}"') for x in os.listdir(data_mask_path / folder)]
    print("Done copy Annotations folder.")
    
def visualize(root):
    import random
    root = Path(root)
    img_path = root / FSS_IMG_DIR
    mask_path = root / FSS_ANNO_DIR
    classes_list = os.listdir(root / FSS_IMG_DIR) 
    classes = random.sample(classes_list, 5 + 1)
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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', help='The folder of dataset')
    parser.add_argument('--dst_folder', help='The destination folder.')
    args = parser.parse_args()

    #pascal2fss(args.data_folder, args.dst_folder)
    #fss1000tofss(args.data_folder, args.dst_folder)
    celeb2fss(args.data_folder, args.dst_folder)
    #visualize(args.dst_folder)
