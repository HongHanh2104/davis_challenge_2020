import torchvision.transforms.functional as F
from PIL import Image
import random

class RandomRotate:
    def __init__(self, rotate_angle):
        assert isinstance(rotate_angle, (list, tuple)) and \
            len(rotate_angle) == 2, \
            f'Invalid type, expect (list, tuple), get {type(rotate_angle)}'
        self.rotate_angle = rotate_angle

    def __call__(self, img, ann):
        start, end = self.rotate_angle
        angle = random.randint(start, end)
        img = F.rotate(img, angle)
        ann = F.rotate(ann, angle)
        return img, ann

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
    rotated_img, rotated_ann = RandomRotate(rotate_angle=(-20, 20))(img, ann)

    print(rotated_img.size, rotated_ann.size)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[0].imshow(ann, alpha=0.5)
    ax[1].imshow(rotated_img)
    ax[1].imshow(rotated_ann, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()

