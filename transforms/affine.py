import torchvision.transforms.functional as F
from torchvision.transforms import RandomAffine
from PIL import Image
import random
import numpy as np

class MultiRandomAffine(RandomAffine):
    
    def __call__(self, frame):
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, frame[0].size)
        return list(map(lambda x: F.affine(x, *ret, resample=self.resample, fillcolor=self.fillcolor), frame))

if __name__ == "__main__":
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help='Path to Image folder.')
    parser.add_argument("--img", help='Image File')
    parser.add_argument("--ann", help='Annotation File')
    args = parser.parse_args()

    img = Image.open(os.path.join(args.root, args.img)).convert('RGB')
    ann = Image.open(os.path.join(args.root, args.ann)).convert('P')
    print(img.size, ann.size)
    affined_img, affined_ann =  MultiRandomAffine(degrees=(-20, 20),
                                                  scale=(0.9, 1.1),
                                                  shear=(-10, 10))((img, ann))
    print(affined_img.size, affined_ann.size)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[0].imshow(ann, alpha=0.5)
    ax[1].imshow(affined_img)
    ax[1].imshow(affined_ann, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()
