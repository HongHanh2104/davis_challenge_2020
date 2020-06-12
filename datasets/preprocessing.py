from pathlib import Path
import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


PASCAL_CLASS = ['aeroplane','bicycle','bird','boat','bottle', \
                'bus','car','cat','chair','cow',\
                'diningtable','dog','horse','motorbike','person', \
                'pottedplant','sheep','sofa','train','tvmonitor']

def create_class_folder(root,
                        seg_class='SegmentationClass',
                        dst_folder='Fss_Pascal'):
    # create class folder
    root = Path(root)
    [os.system('mkdir -p {}/{}/{}'.format(root / dst_folder, x, y)) \
        for x in ['JPEGImages', 'Annotations'] for y in PASCAL_CLASS]

def pascal_preprocess(root,
                      data_folder,
                      seg_class='SegmentationClass',
                      img_folder='JPEGImages',
                      mask_folder='Annotations',
                      dst_folder='Fss_Pascal'):
    # Create folder 
    create_class_folder(root, seg_class, dst_folder)

    root = Path(root)
    root_img_path = root / data_folder / img_folder
    dst_img_path = root / dst_folder / img_folder
    dst_mask_path = root / dst_folder / mask_folder

    for _file in os.listdir(root / data_folder / seg_class):  #['2007_000032.png', '2007_000042.png', '2007_000068.png']: 
        path = root / data_folder / seg_class / _file
        mask = Image.open(path).convert('P')
        class_id = np.unique(mask)
        class_id = [x for x in class_id if 0 < x < 255]
        
        [os.system('cp {}/{} {}/{}'.format(root_img_path, \
                                    _file.replace('png', 'jpg'), \
                                    dst_img_path, \
                                    PASCAL_CLASS[x - 1])) for x in class_id]
        
        
        mask = np.array(mask)
        for _id in class_id:
            copy_mask = np.copy(mask)
            copy_mask[copy_mask != _id] = 0
            save_mask = Image.fromarray(copy_mask).convert('P', palette=Image.ADAPTIVE)
            
            save_mask.save('{}/{}/{}'.format(dst_mask_path, \
                                            PASCAL_CLASS[_id - 1], \
                                            _file))
            
    
    print("Done")

def visualize(root, img_folder, mask_folder):
    import random

    root = Path(root)
    img_path = root / img_folder
    mask_path = root / mask_folder
    classes = random.sample(PASCAL_CLASS, 5 + 1)
    for c in classes:
        imgs = random.sample(os.listdir(img_path / c), 3 + 1)
        print(c)
        for i in range(len(imgs)):
            img = Image.open(img_path / c / imgs[i]).convert('RGB')
            mask = Image.open(mask_path / c / imgs[i].replace('jpg', 'png')).convert('L')
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[1].imshow(mask)
            plt.tight_layout()
            plt.show()
            plt.close()



if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to dataset')
    parser.add_argument('--data_folder', help='The folder of dataset')
    parser.add_argument('--seg_class', help='The folder of SegmentationClass',
                        default='SegmentationClass')
    parser.add_argument('--img_folder', help='The folder of images.',
                        default='JPEGImages')
    parser.add_argument('--mask_folder', help='The folder of masks.',
                        default='Annotations')
    parser.add_argument('--dst_folder', help='The destination folder.', \
                        default='Fss_Pascal')
    args = parser.parse_args()

    #pascal_preprocess(args.root, args.data_folder, args.seg_class, args.img_folder, args.mask_folder, args.dst_folder)
    
    visualize(os.path.join(args.root, args.dst_folder), args.img_folder, args.mask_folder)


