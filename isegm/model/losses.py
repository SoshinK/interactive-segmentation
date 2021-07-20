import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils import misc


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


'''
https://github.com/liewjunhao/thin-object-selection
'''


def binary_cross_entropy_loss(output, label, void_pixels=None,
        class_balance=False, reduction='mean', average='batch'):
    """Binary cross entropy loss for training
    # arguments:
        output: output from the network
        label: ground truth label
        void_pixels: pixels to ignore when computing the loss
        class_balance: to use class-balancing weights
        reduction (str): either 'none'|'sum'|'mean'
            'none': the loss remains the same size as output
            'sum': the loss is summed
            'mean': the loss is average based on the 'average' flag
        average (str): either 'size'|'batch'
            'size': loss divide by #pixels
            'batch': loss divide by batch size
    Remarks: Currently, class_balance=True does not support
    reduction='none'
    """
    assert output.size() == label.size()
    assert not (class_balance and reduction == 'none')

    labels = torch.ge(label, 0.5).float()
    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    
    if class_balance:
        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

        if reduction == 'sum':
            # sum the loss across all elements
            return final_loss
        elif reduction == 'mean':
            # average the loss
            if average == 'size':
                final_loss /= num_total
            elif average == 'batch':
                final_loss /= label.size()[0]
            return final_loss
        else:
            raise ValueError('Unsupported reduction mode: {}'.format(reduction))
    
    else:
        loss_val = -loss_val
        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_val = torch.mul(w_void, loss_val)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
        # final_loss = torch.sum(loss_val)

        if reduction == 'none':
            # return the loss directly
            return loss_val
        elif reduction == 'sum':
            # sum the loss across all elements
            return torch.sum(loss_val)
        elif reduction == 'mean':
            # average the loss
            final_loss = torch.sum(loss_val)
            if average == 'size':
                final_loss /= num_total
            elif average == 'batch':
                final_loss /= label.size()[0]
            return final_loss
        else:
            raise ValueError('Unsupported reduction mode: {}'.format(reduction))


def dice_loss(output, label, void_pixels=None, smooth=1e-8):
    """Dice loss for training.
    Remarks:
    + Sigmoid should be applied before applying this loss.
    + This loss currently only supports average='size'.
    """
    assert output.size() == label.size()
    p2 = (output * output)
    g2 = (label * label)
    pg = (output * label)
    batch_size = output.size(0)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        p2 = torch.mul(p2, w_void)
        g2 = torch.mul(g2, w_void)
        pg = torch.mul(pg, w_void)

    p2 = p2.sum(3).sum(2).sum(1) # (N, )
    g2 = g2.sum(3).sum(2).sum(1) # (N, )
    pg = pg.sum(3).sum(2).sum(1) # (N, )
    final_loss = 1.0 - torch.div((2 * pg), (p2 + g2 + smooth))
    final_loss = torch.sum(final_loss)
    final_loss /= batch_size
    return final_loss

def bootstrapped_cross_entopy_loss(output, label, ratio=1./16, void_pixels=None):
    """Bootstrapped cross-entropy loss used in FRRN 
    <https://arxiv.org/abs/1611.08323>
    Reference:
        [1] Tobias et al. "Full-Resolution Residual Networks for Semantic 
        Segmentation in Street Scenes", CVPR 2017.
        [2] https://github.com/TobyPDE/FRRN/blob/master/dltools/losses.py
    Args:
        output: The output of the network
        label: The ground truth label
        batch_size: The batch size
        ratio: A variable that determines the number of pixels
               selected in the bootstrapping process. The number of pixels
               is determined by size**2 * ratio, where we assume the 
               height and width are the same (size).
    """
    # compute cross entropy
    assert output.size() == label.size()

    labels = torch.ge(label, 0.5).float()
    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    # compute the pixel-wise cross entropy.
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    
    # ignore the void pixels
    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_val = torch.mul(w_void, loss_val)
    
    xentropy = -loss_val

    # for each element in the batch, collect the top K worst predictions
    K = int(label.size(2) * label.size(3) * ratio)
    batch_size = label.size(0)
    
    result = 0.0
    for i in range(batch_size):
        batch_errors = xentropy[i, :]
        flat_errors = torch.flatten(batch_errors)

        # get the worst predictions.
        worst_errors, _ = torch.sort(flat_errors)[-K:]

        result += torch.mean(worst_errors)

    result /= batch_size

    return result

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, void_pixels=None,
        class_balance=False, reduction='mean', average='batch'):
        super().__init__()
        self.void_pixels = void_pixels
        self.class_balance = class_balance
        self.reduction = reduction
        self.average = average

    def forward(self, output, label):
        return binary_cross_entropy_loss(output, label, self.void_pixels, self.class_balance, self.reduction, self.average)

class DiceLoss(nn.Module):
    def __init__(self, void_pixels=None, smooth=1e-8):
        super().__init__()
        self.void_pixels = void_pixels
        self.smooth = smooth
    def forward(self, output, label):
        return dice_loss(output, label, self.void_pixels, self.smooth)

class BootstrappedCrossEntropyLoss(nn.Module):
    def __init__(self, ratio=1./16, void_pixels=None):
        super().__init__()
        self.ratio = ratio
        self.void_pixels = void_pixels
    def forward(self, output, label):
        return bootstrapped_cross_entopy_loss(output, label, self.ratio, self.void_pixels)

class CompositionLosses(nn.Module):
    def __init__(self, criterion_list, weights=None):
        super().__init__()
        self.criterion_list = criterion_list
        self.weights = weights
        if not self.weights is None:
            assert len(self.criterion_list) == len(self.weights) 
        else:
            self.weights = np.ones(len(criterion_list))
    def forward(self, output, label):
        result_loss = 0
        for loss, weight in zip(self.criterion_list, self.weights):
            result_loss += weight * loss(output, label)
        return result_loss