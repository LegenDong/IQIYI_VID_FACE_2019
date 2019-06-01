# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 13:59
# @Author  : LegenDong
# @User    : legendong
# @File    : losses.py
# @Software: PyCharm
import torch
import torch.nn as nn

__all__ = ['FocalLoss', 'FocalLossWithWeight']


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


class FocalLossWithWeight(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithWeight, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target, weight_mask):
        assert target.size() == weight_mask.size()
        loss_unfold = self.ce(x, target)
        weight_mask = torch.mul(loss_unfold, weight_mask)
        log_p = torch.mean(weight_mask)

        p = torch.exp(-log_p)
        loss = (1 - p) ** self.gamma * log_p
        return torch.mean(loss)


if __name__ == '__main__':
    loss_func = FocalLossWithWeight()
    pred = torch.randn(5, 5)
    target = torch.LongTensor([0, 1, 2, 3, 4])
    weight = torch.randn(5)
    loss = loss_func(pred, target, weight)
    print(loss)
