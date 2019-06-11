import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-7
        # only calculate on non-zero pixels in ground_truth
        index_nonzero = torch.nonzero(target).view(-1)
        if index_nonzero.size(0) == 0:
            t = 1
        else:
            self.inter = torch.dot(input.view(-1)[index_nonzero],
                                   target.view(-1)[index_nonzero])
            self.union = torch.sum(input) + torch.sum(target) + eps

            t = (2 * self.inter.float() + eps) / self.union.float()

        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
