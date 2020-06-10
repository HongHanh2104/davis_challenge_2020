import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def __call__(self, output, target):
        target = target.type_as(output)
        if len(target.shape) != len(output.shape):
            target = target.unsqueeze(1)
        return self.loss(output, target)


class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self, beta, **kwargs):
        if isinstance(beta, (float, int)):
            self.beta = torch.Tensor([beta])
        if isinstance(beta, list):
            self.beta = torch.Tensor(beta)
        super().__init__(pos_weight=self.beta, **kwargs)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)


def visualize(batch):
    import matplotlib.pyplot as plt
    *b_annos, c_anno = batch
    n = len(b_annos)

    if len(c_anno.shape) == 4:
        c_anno = torch.argmax(c_anno, dim=1)

    fig, ax = plt.subplots(1, n + 1)

    for i, b_anno in enumerate(b_annos):
        if len(b_anno.shape) == 4:
            b_anno = torch.argmax(b_anno, dim=1)
        ax[i].imshow(b_anno[0].cpu().squeeze(),
                     vmin=0, vmax=11)

    if n == 0:
        ax.imshow(c_anno[0].cpu().squeeze(),
                  vmin=0, vmax=11)
    else:
        ax[n].imshow(c_anno[0].cpu().squeeze(),
                     vmin=0, vmax=11)

    fig.tight_layout()
    plt.show()
    plt.close()


class MultiCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = CrossEntropyLoss(**kwargs)

    def forward(self, output, target):
        # print('Pred')
        # visualize(output)
        # print('Truth')
        # visualize(target)
        w = 1.0 / len(output)
        loss = 0.0
        for pred, true in zip(output, target):
            loss += self.loss(pred, true) * w
        return loss
