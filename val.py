import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as tvtf
from tqdm import tqdm

from datasets import FSSRandomDataset, FSS_ValDataset
from metrics import FSSMeanIoU
from utils.getter import get_instance
from utils.device import move_to
from utils.random_seed import set_seed

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-w', type=str,
                    help='path to weight files')
parser.add_argument('-g', type=int, default=None,
                    help='(single) GPU to use (default: None)')
parser.add_argument('-s', type=int, default=3698,
                    help='random seed (default: 3698)')
parser.add_argument('-f', type=int, default=0,
                    help='fold to validate (default: 0)')

args = parser.parse_args()

# Set seed
set_seed(args.s)

# Device
dev_id = 'cuda:{}'.format(args.g) \
    if torch.cuda.is_available() and args.g is not None \
    else 'cpu'
device = torch.device(dev_id)

# Load model
config = torch.load(args.w, map_location=dev_id)
model = get_instance(config['config']['model']).to(device)
model.load_state_dict(config['model_state_dict'])

# Load data
#dataset = FSSRandomDataset(root='data/PASCAL-5i', 
#			   phase='0_val', 
#                           n=500, 
#                           is_train=False)
dataset = FSS_ValDataset(data_dir='data/PASCAL-5i', 
                         fold=args.f,
                         normalize_mean=[0.485, 0.456, 0.406],
                         normalize_std=[0.229, 0.224, 0.225])
dataloader = DataLoader(dataset, batch_size=1)

# Metrics
metrics = {
    'FSSMeanIoU': FSSMeanIoU(nclasses=5, ignore_index=0)
}

with torch.no_grad():
    for m in metrics.values():
        m.reset()

    model.eval()
    progress_bar = tqdm(dataloader)
    for i, (inp, lbl) in enumerate(progress_bar):
        inp = move_to(inp, device)
        lbl = move_to(lbl, device)
        outs = model(inp)
        for m in metrics.values():
            value = m.calculate(outs, lbl)
            m.update(value)

    print('+ Evaluation result')
    for m in metrics.values():
        m.summary()

