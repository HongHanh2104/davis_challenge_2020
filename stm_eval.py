import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from PIL import Image
import numpy as np
from tqdm import tqdm

from datasets.STM_DAVIS import STM_DAVISTest as DAVIS_MO_Test
from models.stm import STM

import argparse
import os
import random

def overlay_davis(image, mask, colors=[255, 0, 0], cscale=2, alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + \
            np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


torch.set_grad_enabled(False)  # Volatile


def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",
                        default='/local/DATA')
    return parser.parse_args()


args = get_arguments()

GPU = args.g
YEAR = args.y
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on DAVIS')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')


x = os.listdir(DATA_ROOT + '/Annotations/480p/')[0]
palette = Image.open(
    DATA_ROOT + f'/Annotations/480p/{x}/00000.png').getpalette()

@torch.no_grad()
def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i)
                       for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(
            0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    first_frame = Fs[:, :, 0]
    first_mask = Ms[:, :, 0]

    augment_frames = []
    augment_masks = []

    augment_frames.append(torch.flip(first_frame, dims=(-1,)))
    augment_masks.append(torch.flip(first_mask, dims=(-1,)))
    #augment_frames.append(torch.flip(first_frame, dims=(-2,)))
    #augment_masks.append(torch.flip(first_mask, dims=(-2,)))
    augment_size = len(augment_frames)

    keys, values = model(first_frame,
                         first_mask,
                         torch.tensor([num_objects]))
    for aug_f, aug_m in zip(augment_frames, augment_masks):
        new_k, new_v = model(aug_f, aug_m, torch.tensor([num_objects]))
        keys = torch.cat([keys, new_k], dim=3)
        values = torch.cat([values, new_v], dim=3)

    Es = torch.zeros_like(Ms)
    Es[:, :, 0] = Ms[:, :, 0]

    for t in tqdm(range(1, num_frames)):
        if t == 1:
            this_keys, this_values = keys, values  # only prev memory
        else:
            # memorize
            prev_key, prev_value = model(Fs[:, :, t-1], 
                                         Es[:, :, t-1], 
                                         torch.tensor([num_objects]))
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)

        # segment
        logit = model(Fs[:, :, t], 
                      this_keys, this_values,
                      torch.tensor([num_objects]))
        Es[:, :, t] = F.softmax(logit, dim=1)

        for _ in range(1):
            k, v = model(Fs[:, :, t],
                          Es[:, :, t],
                          torch.tensor([num_objects]))
            k = torch.cat([this_keys, k], dim=3)
            v = torch.cat([this_values, v], dim=3)
            logit = model(Fs[:, :, t],
                          k, v,
                          torch.tensor([num_objects]))
            Es[:, :, t] = F.softmax(logit, dim=1)

        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)

    return pred, Es

Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p',
                        imset='20{}/{}.txt'.format(YEAR, SET), single_object=(YEAR == 16))
Testloader = data.DataLoader(
    Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval()  # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = '{}_DAVIS_{}{}'.format(MODEL, YEAR, SET)
print('Start Testing:', code_name)

for seq, V in enumerate(Testloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}, num_objects: {}'.format(
        seq_name, num_frames, num_objects[0][0]))

    pred, Es = Run_video(Fs, Ms, num_frames, num_objects,
                         Mem_every=10, Mem_number=None)

    # Save results for quantitative eval ######################
    test_path = os.path.join('./test', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))

    if VIZ:
        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)

        for f in range(num_frames):
            pF = (Fs[0, :, f].permute(1, 2, 0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))

        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
