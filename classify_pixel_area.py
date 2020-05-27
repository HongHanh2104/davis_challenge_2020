import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def get_canny():
    return 0

def get_labels(img_path, img_name):
    img_path = os.path.join(img_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    binary_obj = []
    for class_id in np.unique(img):
        binary_mask = img == class_id
        binary_obj.append(binary_mask)
    print(len(binary_obj))
    cv2.imshow("test", binary_mask)
    cv2.waitKey()

    blured_img = cv2.medianBlur(img, 7)
    #gray = cv2.bilateralFilter(image, 11, 17, 17)
    #return cv2.Canny(gray, 30, 200)
    ret, labels = cv2.connectedComponents(blured_img)
    
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return ret - 1, labeled_img

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='path to the root')
    parser.add_argument('--anno', help='path to annotation folder',
                        default='test')
    parser.add_argument('--img', help='name of image')
    args = parser.parse_args()

    img_path = os.path.join(args.root, args.anno)
    numbers, image = get_labels(img_path, args.img)
    #print(type(image), image.shape)
    print(numbers)
    plt.imshow(image)
    plt.show()
    #cv2.imshow("test", image)
    #cv2.waitKey()

if __name__ == "__main__":
    test()