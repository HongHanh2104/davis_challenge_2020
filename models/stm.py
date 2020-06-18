import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import math


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Conv2d(indim, outdim, 
                                        kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, 
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, 
                               kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        x = self.downsample(x)
        return x + r


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m):
        #f = (in_f - self.mean) / self.std
        f = in_f
        
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim

        x = self.conv1(f) + self.conv1_m(m)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        #f = (in_f - self.mean) / self.std
        f = in_f

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, 
                                kernel_size=3, padding=1, stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor,
                              mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, 
                                kernel_size=3, padding=1, stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, 
                               kernel_size=3, padding=1, stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4,
                          mode='bilinear', align_corners=False)
        return p  # , p2, p3, p4


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, Hi, Wi = m_in.size()
        _,   _, Ho, Wo = q_in.size()
        _, D_o,  _,  _ = m_out.size()

        mi = m_in.reshape(B, D_e, Hi*Wi)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.reshape(B, D_e, Ho*Wo)  # b, emb, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_out.reshape(B, D_o, Hi*Wi)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.reshape(B, D_o, Ho, Wo)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim,
                             kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim,
                                kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)

    def memorize(self, frame, mask):
        r4, _, _, _, _ = self.Encoder_M(frame, mask)
        k4, v4 = self.KV_M_r4(r4)
        return k4, v4

    def segment(self, frame, keys, values):
        _, _, H, W = keys.shape  # B = 1
        [frame], pad = pad_divide_by([frame], 16, 
                                     (frame.size(2), frame.size(3)))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        m4, viz = self.Memory(keys, values, k4, v4)

        logit = self.Decoder(m4, r3, r2)

        if pad[2]+pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]
        if pad[0]+pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        return logit


def visualize(batch):
    import matplotlib.pyplot as plt
    import numpy as np

    ref_imgs, ref_masks, q_img = batch
    k = len(ref_imgs)

    fig, ax = plt.subplots(1, k + 1)

    for i, (ref_img, ref_mask) in enumerate(zip(ref_imgs, ref_masks)):
        ax[i].imshow(ref_img[0].cpu().permute(1, 2, 0))
        ax[i].imshow(ref_mask[0].cpu().squeeze(0), alpha=0.5)

    ax[k].imshow(q_img[0].cpu().permute(1, 2, 0))

    fig.tight_layout()
    plt.show()
    plt.close()


class STMOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        stm = nn.DataParallel(STM())
        # stm.load_state_dict(torch.load('STM_weights.pth'))
        self.stm = stm.module

    def forward(self, inp):
        # visualize(inp)
        # self.stm.eval()

        ref_imgs, ref_masks, q_img = inp

        # Memorize the first reference image
        k, v = self.stm.memorize(ref_imgs[0], ref_masks[0])

        # Memorize the rest of the reference images
        for i in range(1, len(ref_imgs) - 1):
            nk, nv = self.stm.memorize(ref_imgs[i], ref_masks[i])
            k = torch.cat([k, nk], dim=3)
            v = torch.cat([v, nv], dim=3)

        s_logits = [self.stm.segment(ref_img, k, v) for ref_img in ref_imgs]
        q_logit = self.stm.segment(q_img, k, v)

        return (*s_logits, q_logit)
