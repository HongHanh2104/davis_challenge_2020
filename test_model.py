import torch 
from models.canet import CANet
from models.resnet import ResNetExtractor
from PIL import Image
import numpy as np
from torchvision import transforms as tvtf

class NormMaxMin():
    def __call__(self, x):
        return (x.float() - torch.min(x)) / (torch.max(x) - torch.min(x))

def getImage(support_path, annotation_path, query_path):
    support_img = Image.open(support_path).convert('RGB')
    support_arr = np.array(support_img)
    support_img_tf = tvtf.Compose([
        NormMaxMin()
    ])  #np.array(support_name)
    support_img = support_img_tf(torch.Tensor(support_arr)).unsqueeze(0).permute(0, 3, 1, 2)
    
    query_img = Image.open(query_path).convert('RGB')
    query_arr = np.array(query_img)
    query_img_tf = tvtf.Compose([
        NormMaxMin()
    ])
    query_img = query_img_tf(torch.Tensor(query_arr)).unsqueeze(0).permute(0, 3, 1, 2)
    
    anno_img = Image.open(annotation_path).convert("L")
    anno_arr = np.array(anno_img)
    anno_img_tf = tvtf.Compose([
    ])
    anno_img = torch.Tensor(anno_img_tf(anno_arr)).unsqueeze(0).unsqueeze(0)

    return support_img, anno_img, query_img    

def test(support_path, anno_path, query_path):
    '''
    query = torch.rand(1,3,854,480).cpu()
    support = torch.rand(1,3,854,480).cpu()
    annotation = torch.rand(1,1,854,480).cpu()
    '''
    support_img, anno_img, query_img = getImage(support_path, anno_path, query_path)
    print(support_img.shape, anno_img.shape, query_img.shape)
    extractor = ResNetExtractor('resnet50').cpu()
    net = CANet(num_class=2, extractor=extractor).cpu()
    out = net(support=support_img, annotation=anno_img, query=query_img)

    return torch.argmax(out, dim=1)

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--support")
    parser.add_argument("--annotation")
    parser.add_argument("--query")
    args = parser.parse_args()
    pred = test(args.support, args.annotation, args.query)
    #pred = pred.detach()
    print(pred.shape)
    plt.imshow(pred.squeeze(0))
    plt.show()
    plt.close()
    