import os
import sys
import argparse

from PIL import Image
import numpy as np
from tqdm import tqdm

# Get root DAVIS folder
parser = argparse.ArgumentParser()
parser.add_argument('--root', help='path to DAVIS-like folder')
parser.add_argument('--copy', action='store_true', help='copy mode (create duplicate folders), default: symlink mode (create symbolic links)')
args = parser.parse_args()

data_dir = args.root
dup_cmd = 'cp -r' if args.copy else 'ln -s'

# Annotations dir (reference, split, all)
anno_dir = os.path.join(data_dir, 'Annotations/480p/')
save_dir = os.path.join(data_dir, 'Annotations/480p_split/')
save_dir_all = os.path.join(data_dir, 'Annotations/480p_all/')
# JPEG dir (reference, split, all)
jpeg_dir = anno_dir.replace('Annotations', 'JPEGImages')
save_jpeg_dir = save_dir.replace('Annotations', 'JPEGImages')
save_jpeg_dir_all = save_dir_all.replace('Annotations', 'JPEGImages')
os.makedirs(save_jpeg_dir, exist_ok=True)
os.makedirs(save_jpeg_dir_all, exist_ok=True)

fds = os.listdir(anno_dir)
for fd in tqdm(fds):
    im_list = os.listdir(os.path.join(anno_dir, fd))
    im_list = [item for item in im_list if item[-3:] == 'png']
    first_img = sorted(os.listdir(anno_dir + fd))[0]
    im = np.array(Image.open(os.path.join(anno_dir + fd, first_img)))
    cls_n = im.max()

    # Create symlinks for JPEGImages (fake multiple videos)
    cmd = f"{dup_cmd} {os.path.abspath(f'{jpeg_dir}/{fd}')} {save_jpeg_dir_all}"
    os.system(cmd)
    for i in range(1, cls_n+1):
        if not os.path.exists(f'{save_jpeg_dir}/{fd}_{i}'):
            cmd = f"{dup_cmd} {os.path.abspath(f'{jpeg_dir}/{fd}')} {save_jpeg_dir}/{fd}_{i}"
            os.system(cmd)

    for item in im_list:
        im_path = os.path.join(anno_dir, fd, item)
        im = np.array(Image.open(im_path))
        all_dir = os.path.join(save_dir_all, fd)
        os.makedirs(all_dir, exist_ok=True)
        binary_map = (im > 0)
        mask_image = Image.fromarray((binary_map).astype(np.uint8))
        mask_image.save(os.path.join(all_dir, item))

        for i in range(1, cls_n+1):
            split_dir = f'{save_dir}/{fd}_{i}'
            os.makedirs(split_dir, exist_ok=True)
            binary_map = im == i
            mask_image = Image.fromarray((binary_map).astype(np.uint8))
            mask_image.save(os.path.join(split_dir, item))
