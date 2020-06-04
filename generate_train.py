import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from models.stm import STM
from datasets.davis import DAVISTripletDataset
from utils.device import move_to
from utils.random_seed import set_seed, set_determinism

set_seed(3698)
set_determinism()

# Load model
model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval()

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))
model = model.module

# Load dataset
dataset = DAVISTripletDataset(root_path='data/DAVIS-trainval',
                              resolution='480p',
                              phase='train',
                              mode=4,
                              is_train=False)
dataloader = DataLoader(dataset, batch_size=1)

# Metric


def iou(output, target, nclasses, eps=1e-9):
    batch_size = output.size(0)
    ious = torch.zeros(nclasses, batch_size)

    prediction = torch.argmax(output, dim=1)
    prediction = F.one_hot(prediction, nclasses).bool()
    target = F.one_hot(target, nclasses).bool()
    intersection = (prediction & target).sum((-3, -2))
    union = (prediction | target).sum((-3, -2))
    ious = (intersection.float() + eps) / (union.float() + eps)
    # ious = ious[:, 1:]

    return ious.mean()

f = open('train.csv', 'w')
with torch.no_grad():
    for idx, (inp, out) in enumerate(tqdm(dataloader)):
        a_img, a_anno, b_img, c_img, infos = move_to(
            inp, torch.device('cuda:0'))
        b_anno, c_anno = move_to(out, torch.device('cuda:0'))
        
        a_id, b_id, c_id = infos['frames']
        a_id = a_id[0]
        b_id = b_id[0]
        c_id = c_id[0]

        nobjects = infos['nobjects'].item()
        num_objects = torch.LongTensor([[nobjects]])

        # Memorize a
        a_seg = F.one_hot(a_anno, 11).permute(0, 3, 1, 2)
        k, v = model.memorize(a_img, a_seg, num_objects)

        # Without using intermediate frame
        logit = model.segment(c_img, k, v, num_objects)
        iou_1 = iou(logit, c_anno, 1 + nobjects).item()

        # Using intermediate frame
        # Segment b
        b_logit = model.segment(b_img, k, v, num_objects)
        # Memorize b
        b_pred = F.softmax(b_logit, dim=1)
        #b_im = self.stn(b_im, b_pred)
        b_k, b_v = model.memorize(b_img, b_pred, num_objects)
        k = torch.cat([k, b_k], dim=3)
        v = torch.cat([v, b_v], dim=3)

        logit = model.segment(c_img, k, v, num_objects)
        iou_2 = iou(logit, c_anno, 1 + nobjects).item()

        f.write(','.join([a_id, b_id, c_id, str(iou_1), str(iou_2)]) + '\n')

        # fig, ax = plt.subplots(3, 1)
        # ax[0].imshow(a_img[0].permute(1, 2, 0))
        # ax[0].imshow(a_anno[0].squeeze(0),
        #              vmin=0, vmax=nobjects, alpha=0.5)
        # ax[1].imshow(b_img[0].permute(1, 2, 0))
        # ax[1].imshow(b_anno[0].squeeze(0),
        #              vmin=0, vmax=nobjects, alpha=0.5)
        # ax[2].imshow(c_img[0].permute(1, 2, 0))
        # ax[2].imshow(c_anno[0].squeeze(0),
        #              vmin=0, vmax=nobjects, alpha=0.5)

        # fig.tight_layout()
        # plt.show()
        # plt.close()
f.close()
