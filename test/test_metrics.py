import torch 
from metrics.segmentation.mean_iou import MeanIoU 

if __name__ == "__main__":
    nclasses = 3
    ignore_index = -1

    target = torch.Tensor([
        [
            [1, 0],
            [2, 1]
        ],
        [
            [1, 2],
            [2, 1]
        ],
        [
            [1, 1],
            [1, 1]
        ]
    ]).long()

    output = torch.Tensor([
        [
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ],
        [
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ],
        [
            [[1, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ]
    ]).float()

    metric = MeanIoU(nclasses=nclasses, ignore_index=-1)
    print(metric.calculate(output, target))