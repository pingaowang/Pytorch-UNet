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


def iou(truth, pred):
    in_non_zero_truth = np.where(truth != 0)
    truth_1 = truth[in_non_zero_truth]
    pred_1 = pred[in_non_zero_truth]
    intersection = np.logical_and(truth_1, pred_1)
    union = np.logical_or(truth_1, pred_1)
    return np.sum(intersection) / (np.sum(union) + 1e-7)


def iou_all(truth, pred):
    intersection = np.logical_and(truth, pred)
    union = np.logical_or(truth, pred)
    return (np.sum(intersection) + 5e-7) / (np.sum(union) + 1e-6)



