import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop
from PIL import Image
import random

class MyRandomCrop:
    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, float, list, tuple)), \
            f'Invalid type, expect (int, float, list, tuple), get {type(crop_size)}'
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if isinstance(crop_size, float):
            assert 0 < crop_size <= 1, \
                f'Invalid crop size, float should be in (0, 1], get {crop_size}'
            crop_size = (crop_size, crop_size)
        if isinstance(crop_size, (list, tuple)):
            crop_size = crop_sizes
        self.crop_size = crop_size

    def __call__(self, x):
        # x => C, H, W
        h, w = x.size()[-2:]

        crop_size = (min(self.crop_size[0], h),
                     min(self.crop_size[1], w))
        r0 = torch.randint(high=h - crop_size[0] + 1, size=(1,))
        c0 = torch.randint(high=w - crop_size[1] + 1, size=(1,))
        r1 = r0 + crop_size[0]
        c1 = c0 + crop_size[1]

        return r0, c0, r1, c1

   
class MultiRandomCrop(RandomCrop):
    def __call__(self, frame):
        i, j, h, w = self.get_params(frame[0], self.size)
        return list(map(lambda x: F.crop(x, i, j, h, w), frame))

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
    #print(img.size, ann.size)
    
    croped_img, croped_ann =  MultiRandomCrop(size=384)((img,ann))

    #print(croped_img.size, croped_ann.size)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[0].imshow(ann, alpha=0.5)
    ax[1].imshow(croped_img)
    ax[1].imshow(croped_ann, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()

