import torchvision.transforms as tvtf
from PIL import Image
import argparse
from utils.device import move_to
from utils.getter import get_instance, get_data, get_single_data
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 5)


parser = argparse.ArgumentParser()
parser.add_argument('--weight')
parser.add_argument('--gpus', default=None)
parser.add_argument('--ref_img', default=None)
parser.add_argument('--ref_mask', default=None)
parser.add_argument('--query_img', default=None)
args = parser.parse_args()

dev_id = 'cuda:{}'.format(args.gpus) \
    if torch.cuda.is_available() and args.gpus is not None \
    else 'cpu'
device = torch.device(dev_id)

config = torch.load(args.weight, map_location=dev_id)

model = get_instance(config['config']['model']).to(device)
# model.load_state_dict(config['model_state_dict'])
# model.stm = nn.DataParallel(model.stm)
# model.stm.load_state_dict(torch.load('STM_weights.pth'))
# model.stm = model.stm.module

print(config['config'])

with torch.no_grad():
    ref_img = Image.open(args.ref_img).convert('RGB')
    ref_mask = Image.open(args.ref_mask).convert('P')
    query_img = Image.open(args.query_img).convert('RGB')

    ref_img = tvtf.ToTensor()(ref_img)
    ref_mask = torch.LongTensor(np.array(ref_mask) > 0)
    query_img = tvtf.ToTensor()(query_img)

    out = model.predict(move_to(([ref_img.unsqueeze(0)],
                                 [ref_mask.unsqueeze(0)],
                                 query_img.unsqueeze(0)),
                                device))
    pred = torch.argmax(out, dim=1).detach().cpu()[0]

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(ref_img.permute(1, 2, 0))
    ax[0].imshow(ref_mask, alpha=0.5)
    ax[0].set_title('Reference')

    ax[1].imshow(query_img.permute(1, 2, 0))
    ax[1].imshow(pred.squeeze(0), alpha=0.5)
    ax[1].set_title('Prediction')

    fig.tight_layout()
    plt.show()
    plt.close()
