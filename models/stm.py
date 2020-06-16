import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import math


def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda()
    else:
        return xs


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
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(
            indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7,
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

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()  # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o)
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
        f = (in_f - self.mean) / self.std

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
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(
            3, 3), padding=(1, 1), stride=1)
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
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(
            3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(
            3, 3), padding=(1, 1), stride=1)

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
        # print(m_in.shape, m_out.shape, q_in.shape, q_out.shape)
        B, D_e, T, Hi, Wi = m_in.size()
        _, _, Ho, Wo = q_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*Hi*Wi)
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb

        qi = q_in.view(B, D_e, Ho*Wo)  # b, emb, HW

        p = torch.bmm(mi, qi)  # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)  # b, THW, HW

        mo = m_out.view(B, D_o, T*Hi*Wi)
        mem = torch.bmm(mo, p)  # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, Ho, Wo)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(
            3, 3), padding=(1, 1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(
            3, 3), padding=(1, 1), stride=1)

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

    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = torch.zeros(1, K, mem.size()[1],
                                  1, mem.size()[2], mem.size()[3]).to(mem.device)
            pad_mem[0, 1:num_objects+1, :, 0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects):
        # memorize a frame
        num_objects = num_objects[0].item()
        _, K, H, W = masks.shape  # B = 1

        (frame, masks), pad = pad_divide_by(
            [frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f': [], 'm': [], 'o': []}
        for o in range(1, num_objects+1):  # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:, o])
            B_list['o'].append((torch.sum(masks[:, 1:o], dim=1) +
                                torch.sum(masks[:, o+1:num_objects+1], dim=1)).clamp(0, 1))

        # make Batch
        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        k4, v4 = self.KV_M_r4(r4)  # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = torch.zeros(1, K, H, W).to(ps.device)
        em[0, 0] = torch.prod(1-ps, dim=0)  # bg prob
        em[0, 1:num_objects+1] = ps  # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em / (1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects):
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape  # B = 1
        # pad
        [frame], pad = pad_divide_by(
            [frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16

        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects, -1, -1, -
                             1), v4.expand(num_objects, -1, -1, -1)
        r3e, r2e = r3.expand(num_objects, -1, -1, -
                             1), r2.expand(num_objects, -1, -1, -1)

        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0, 1:num_objects+1],
                              values[0, 1:num_objects+1], k4e, v4e)

        # import matplotlib.pyplot as plt
        # for i in range(viz.size(2)):
        #     plt.subplot(1, 2, 1)
        #     m = torch.zeros(k4e.shape[-2], k4e.shape[-1])
        #     m[i // k4e.shape[-2], i % k4e.shape[-2]] = 1
        #     m = F.interpolate(m.unsqueeze(0).unsqueeze(
        #         0), (frame.shape[-2], frame.shape[-1])).squeeze(0).squeeze(0)
        #     f = frame[0].permute(1, 2, 0).detach().cpu()
        #     plt.imshow(f)
        #     plt.imshow(m, alpha=0.5)
        #     plt.subplot(1, 2, 2)
        #     v = viz[0, :, i].reshape(14, 14).detach().cpu()
        #     v = F.interpolate(v.unsqueeze(
        #         0).unsqueeze(0), (224, 224)).squeeze(0).squeeze(0)
        #     plt.imshow(v)
        #     plt.tight_layout()
        #     plt.savefig(f'viz_/{i}')
        #     # plt.show()
        #     plt.close()

        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:, 1]  # no, h, w
        # ps = indipendant possibility to belong to each object

        logit = self.Soft_aggregation(ps, K)  # 1, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]
        if pad[0]+pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        return logit

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4:  # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)


def visualize(batch):
    import matplotlib.pyplot as plt
    import numpy as np

    a_img, a_anno, *b_imgs, c_img, nobjects = batch
    n = len(b_imgs)

    fig, ax = plt.subplots(1, n + 2)
    ax[0].imshow(a_img[0].cpu().permute(1, 2, 0))
    ax[0].imshow(a_anno[0].cpu().squeeze(),
                 vmin=0, vmax=nobjects, alpha=0.5)
    print(np.unique(a_anno.cpu()))

    for i, b_img in enumerate(b_imgs):
        ax[1+i].imshow(b_img[0].cpu().permute(1, 2, 0))

    ax[n+1].imshow(c_img[0].cpu().permute(1, 2, 0))

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
        self.stm.eval()

        ref_imgs, ref_masks, q_img = inp
        num_objects = torch.LongTensor([1])

        # Memorize the first reference image
        first_img = ref_imgs[0]
        first_mask = F.one_hot(ref_masks[0], 2).permute(0, 3, 1, 2)
        k, v = self.stm.memorize(first_img, first_mask, num_objects)

        # Memorize the rest of the reference images
        for i in range(1, len(ref_imgs) - 1):
            img = ref_imgs[i]
            mask = F.one_hot(ref_masks[i], 2).permute(0, 3, 1, 2)
            nk, nv = self.stm.memorize(img, mask, num_objects)
            k = torch.cat([k, nk], dim=3)
            v = torch.cat([v, nv], dim=3)

        logit = self.stm.segment(q_img, k, v, num_objects)

        return logit
