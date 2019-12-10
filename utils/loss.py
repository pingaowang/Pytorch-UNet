import torch
import numpy as np

SMOOTH = 1e-6

def iou_loss(outputs: torch.Tensor, labels: torch.Tensor):
    in_nonzero = torch.nonzero(labels)
    outputs = outputs[in_nonzero]
    labels = labels[in_nonzero]

    intersection = (outputs & labels)
    union = (outputs | labels)

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou




