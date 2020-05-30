import torchvision.transforms.functional as F

from PIL import Image
import random

class MultiRandomResize:
    def __init__(self, resize_value):
        assert isinstance(resize_value, int)
        self.resize_value = resize_value
    
    def __call__(self, frame):
        w, h = frame[0].size

        to_size = min(w, h)
        if to_size <= self.resize_value:
            random_value = self.resize_value
        else:
            random_value = random.randint(self.resize_value, to_size)
        #min_size = min(self.resize_value, to_size)
        #max_size = max(self.resize_value, to_size)
        #random_value = max(self.resize_value, random.randint(min_size, max_size))
        
        img = F.resize(frame[0], 
                       size=random_value, 
                       interpolation=Image.BILINEAR)
        ann = F.resize(frame[1],
                       size=random_value,
                       interpolation=Image.NEAREST)
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
    resized_img, resized_ann = MultiRandomResize(resize_value=384)((img, ann))
    print(type(resized_img))
    print(resized_img.size, resized_ann.size)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[0].imshow(ann, alpha=0.5)
    ax[1].imshow(resized_img)
    ax[1].imshow(resized_ann, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.close()

