# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 13:59
# @Author  : LegenDong
# @User    : legendong
# @File    : losses.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.autograd.function import Function as F
from torch.nn import Parameter


__all__ = ['FocalLoss', 'CenterLoss', 'GitLoss', 'RingLoss']


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x, target):
        log_p = self.ce(x, target)
        p = torch.exp(-log_p)
        loss = (1 - p) ** self.gamma * log_p
        return torch.mean(loss)


class CenterLoss(nn.Module):
    """
    Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        """
        :param num_classes: number of classes
        :param feat_dim: dimension of feature
        :param use_gpu: if use gpu or not
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        :param x: feature matrix with shape (batch_size, feat_dim).
        :param labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class RingLoss(nn.Module):
    """
    Ring Loss

    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018
    """

    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        """
        :param x: feature matrix
        :return: ring loss
        """
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0:
            # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1':
            # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.cuda().expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.cuda().expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto':
            # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.cuda().expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else:
            # L2 Loss, if not specified
            diff = x.sub(self.radius.cuda().expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss


class GitLoss(nn.Module):
    """Git loss.

        Reference:
        Calefati et al. Git loss for deep face recognition.

        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
            lambda_c (float): super parameter
            lambda_g (float): super parameter
        """
    def __init__(self, num_classes=10035, feat_dim=512, lambda_c=1, lambda_g=1):
        super(GitLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lamda_c = lambda_c
        self.lamda_g = lambda_g

    def forward(self, x, labels):

        # todo
        pass
