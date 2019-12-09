import numpy as np


def iou(truth, pred):
    in_non_zero_truth = np.where(truth != 0)
    truth_1 = truth[in_non_zero_truth]
    pred_1 = pred[in_non_zero_truth]
    intersection = np.logical_and(truth_1, pred_1)
    union = np.logical_or(truth_1, pred_1)
    return np.sum(intersection) / (np.sum(union) + 1e-7)




