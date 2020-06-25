import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import math


def get_pad_array(size, d):
    h, w = size
    dh, dw = 0, 0
    if h % d > 0:
        dh = d - h % d
    if w % d > 0:
        dw = d - w % d
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    return (left, right, top, bottom)


def pad_divide_by(inp, d):
    pad_array = get_pad_array(inp.shape[-2:], d)
    padded = F.pad(inp, pad_array)
    return padded, pad_array


class ResBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.conv1 = nn.Conv2d(indim, outdim,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim,
                               kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        return x + r


class Encoder_M(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool

        self.res2 = encoder.layer1
        self.res3 = encoder.layer2
        self.res4 = encoder.layer3

    def forward(self, f, m):
        # f: B, C, H, W
        # m: B, H, W
        m = m.unsqueeze(1).float()  # m: B, 1, H, W
        x = self.conv1(f) + self.conv1_m(m)  # x: B, D/8, H/2, W/2
        x = self.bn1(x)  # x: B, D/8, H/2, W/2
        c1 = self.relu(x)  # c1: B, D/8, H/2, W/2
        x = self.maxpool(c1)  # x: B, D/8, H/4, W/4
        r2 = self.res2(x)  # r2: B, D/4, H/4, W/4
        r3 = self.res3(r2)  # r3: B, D/2, H/8, W/8
        r4 = self.res4(r3)  # r4: B, D, H/16, W/16
        return r4, r3, r2, c1, f


class Encoder_Q(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool

        self.res2 = encoder.layer1
        self.res3 = encoder.layer2
        self.res4 = encoder.layer3

    def forward(self, f):
        # f: B, C, H, W
        x = self.conv1(f)  # x: B, D/8, H/2, W/2
        x = self.bn1(x)  # x: B, D/8, H/2, W/2
        c1 = self.relu(x)  # c1: B, D/8, H/2, W/2
        x = self.maxpool(c1)  # x: B, D/8, H/4, W/4
        r2 = self.res2(x)  # r2: B, D/4, H/4, W/4
        r3 = self.res3(r2)  # r3: B, D/2, H/8, W/8
        r4 = self.res4(r3)  # r4: B, D, H/16, W/16
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
        # I = inplanes, O = planes, S = scale factor
        # f: B, I, H*S, W*S
        # pm: B, O, H, W
        s = self.convFS(f)  # s: B, O, HS, WS
        s = self.ResFS(s)  # s: B, O, HS, WS
        pm = F.interpolate(pm, scale_factor=self.scale_factor,
                           mode='bilinear', align_corners=False)
        # pm: B, O, HS, WS
        m = s + pm  # m: B, O, HS, WS
        m = self.ResMM(m)  # m: B, O, HS, WS
        return m


class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim,
                                kernel_size=3, padding=1, stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)
        self.RF2 = Refine(256, mdim)

        self.pred2 = nn.Conv2d(mdim, 2,
                               kernel_size=1, padding=0, stride=1)

    def forward(self, r4, r3, r2):
        # r2: B, D/4, H/4, W/4
        # r3: B, D/2, H/8, W/8
        # r4: B, D, H/16, W/16

        m4 = self.convFM(r4)  # m4: B, D/4, H/16, W/16
        m4 = self.ResMM(m4)  # m4: B, D/4, H/16, W/16
        m3 = self.RF3(r3, m4)  # m3: B, D/4, H/8, W/8
        m2 = self.RF2(r2, m3)  # m2: B, D/4, H/4, W/4

        p2 = F.relu(m2)  # p2: B, D/4, H/4, W/4
        p2 = self.pred2(p2)  # p2: B, N, H/4, W/4

        p = F.interpolate(p2, scale_factor=4,
                          mode='bilinear', align_corners=False)
        # p: B, N, H, W
        return p  # , p2, p3, p4

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.attn = nn.MultiheadAttention(512, 4)

    def forward(self, m_k, m_v, q_k):
        # m_k: B, Dk, Hm, Wm
        # m_v: B, Dv, Hm, Wm
        # q_k: B, Dk, Hq, Wq

        B, Dk, Hm, Wm = m_k.size()
        _,  _, Hq, Wq = q_k.size()
        _, Dv,  _,  _ = m_v.size()

        mk = m_k.reshape(B, Dk, Hm*Wm)  # mk: B, D, Hm*Wm
        mk = mk.permute(2, 0, 1)  # mk: Hm*Wm, B, D

        qk = q_k.reshape(B, Dk, Hq*Wq)  # qk: B, D, Hq*Wq
        qk = qk.permute(2, 0, 1) # qk: Hq*Wq, B, D

        mv = m_v.reshape(B, Dv, Hm*Wm)  # mv: B, D, Hm*Wm
        mv = mv.permute(2, 0, 1) # mv: Hm*Wm, D, B

        mem, p = self.attn(qk, mk, mv) 
        # mem: Hq*Wq, B, D
        # p: B, Hq*Wq, Hm*Wm
        mem = mem.permute(1, 2, 0) # mem: B, D, Hq*Wq
        mem = mem.reshape(B, Dk, Hq, Wq)

        return mem, p

class _Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_k, m_v, q_k):
        # m_k: B, Dk, Hm, Wm
        # m_v: B, Dv, Hm, Wm
        # q_k: B, Dk, Hq, Wq

        B, Dk, Hm, Wm = m_k.size()
        _,  _, Hq, Wq = q_k.size()
        _, Dv,  _,  _ = m_v.size()

        mk = m_k.reshape(B, Dk, Hm*Wm)  # mk: B, Dk, Hm*Wm
        mk = torch.transpose(mk, 1, 2)  # mk: B, Hm*Wm, Dk

        qk = q_k.reshape(B, Dk, Hq*Wq)  # qk: B, Dk, Hq*Wq

        p = torch.bmm(mk, qk)  # p: B, Hm*Wm, Hq*Wq
        p = p / math.sqrt(Dk)  # p: B, Hm*Wm, Hq*Wq
        p = F.softmax(p, dim=1)  # p: B, Hm*Wm, Hq*Wq

        mv = m_v.reshape(B, Dv, Hm*Wm)  # mv: B, Dv, Hm*Wm
        mem = torch.bmm(mv, p)  # B, Dv, Hq*Wq
        mem = mem.reshape(B, Dv, Hq, Wq)  # B, Dv, Hq, Wq

        return mem, p

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim,
                             kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim,
                               kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        with torch.no_grad():
            pe = positionalencoding2d(*x.shape[-3:]).to(x.device)
            x = x + pe
        # x: B, D, H/16, W/16
        k = self.Key(x)  # k: B, K, H/16, W/16
        v = self.Value(x)  # v: B, V, H/16, W/16
        return k, v


class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.backbone_M = models.resnet50(pretrained=True)
        self.backbone_Q = models.resnet50(pretrained=True)

        #for p in self.backbone.parameters():
        #    p.requires_grad = False

        self.Encoder_M = Encoder_M(self.backbone_M)
        self.Encoder_Q = Encoder_Q(self.backbone_Q)

        self.KV_M_r4 = KeyValue(1024, keydim=512, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=512, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)

    def memorize(self, frame, mask):
        # frame: B, C, H, W
        # mask: B, H, W
        r4, _, _, _, _ = self.Encoder_M(frame, mask)
        # r4: B, D, H/16, W/16
        k4, v4 = self.KV_M_r4(r4)
        # k4: B, Dk, H/16, W/16
        # v4: B, Dv, H/16, W/16
        return k4, v4

    def segment(self, frame, keys, values):
        # frame: B, C, H, W
        # keys: B, Dk, Hm, Wm
        # values: B, Dv, Hm, Wm

        # Pad frame for consistent rescaling
        frame, pad = pad_divide_by(frame, 16)
        # frame: B, C, H, W (padded)

        # Encode query frame as a (key, value) pair
        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        # r2: B, D/4, H/4, W/4
        # r3: B, D/2, H/8, W/8
        # r4: B, D, H/16, W/16
        k4, v4 = self.KV_Q_r4(r4)
        # k4: B, Dk, H/16, W/16
        # v4: B, Dv, H/16, W/16

        # Read from memory
        mem, _ = self.Memory(keys, values, k4)
        m4 = torch.cat([mem, v4], dim=1)
        # m4: B, D, H/16, W/16

        # Decode
        logit = self.Decoder(m4, r3, r2)
        # logit: B, N, H, W

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]
        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]
        # logit: B, N, H, W (original)

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
        #self.load_state_dict(torch.load('backup/stm_best.pth'))

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
