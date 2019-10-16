import torch
import torch.nn.functional as F
import numpy as np

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    i_out = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        true_mask = np.transpose(true_mask, (0, 3, 1, 2))
        true_mask = true_mask.float()

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()
        mask_pred = mask_pred.unsqueeze(0)

        tot += dice_coeff(mask_pred, true_mask).item()
        i_out = i
    return tot / (i_out + 1)
