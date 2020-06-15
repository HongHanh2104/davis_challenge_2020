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

IMAGE_FOLDER = "JPEGImages"
MASK_FOLDER = "Annotations"
IMAGE_SETS = "ImageSets"
SEGCLASS_FOLDER = "SegmentationClass"

def create_data_structure(root,
                        classes,
                        dst_folder):
    #[os.system(f"mkdir -p {root / dst_folder}/JPEGImages/{y}") for y in classes[0:5]]
    
    # Create ImageSets folder
    os.system(f"mkdir -p {root / dst_folder}/{IMAGE_SETS}")

    # Create JPEGImages and Annotations
    [os.system(f"mkdir -p {root / dst_folder}/{x}/{y}") \
        for x in [IMAGE_FOLDER, MASK_FOLDER] for y in classes]

def create_class_txt(root,
                     classes,
                     dst_folder):
    
    namefile = "classes.txt"
    with open(root / dst_folder / "ImageSets" / namefile, "w") as f:
        for c in classes:
            f.write(c + "\n")        

def pascal_preprocess(root,
                      data_folder,
                      dst_folder='fss_pascal'):
    
    root = Path(root)

    # Create folder 
    create_data_structure(root, PASCAL_CLASS, dst_folder)

    # Create class txt file
    create_class_txt(root, PASCAL_CLASS, dst_folder)
    
    root_img_path = root / data_folder / IMAGE_FOLDER
    segclass_path = root / data_folder / SEGCLASS_FOLDER
    dst_img_path = root / dst_folder / IMAGE_FOLDER
    dst_mask_path = root / dst_folder / MASK_FOLDER
        
    for _file in os.listdir(segclass_path):  #['2007_000032.png', '2007_000042.png', '2007_000068.png']: 
        path = segclass_path / _file
        mask = Image.open(path).convert('P')
        class_id = np.unique(mask)
        class_id = [x for x in class_id if 0 < x < 255]
        # Copy JPEGImages into class folders
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
            mask = Image.open(mask_path / c / imgs[i].replace('jpg', 'png')).convert('L')
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
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to dataset')
    parser.add_argument('--data_folder', help='The folder of dataset')
    parser.add_argument('--dst_folder', help='The destination folder.')
    args = parser.parse_args()

    #pascal_preprocess(args.root, args.data_folder, args.dst_folder)
    
    visualize(os.path.join(args.root, args.dst_folder))

    #fss1000_preprocess(args.root, args.data_folder, args.img_folder, args.mask_folder, args.dst_folder)


