# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 13:59
# @Author  : LegenDong
# @User    : legendong
# @File    : losses.py
# @Software: PyCharm
import torch
import torch.nn as nn

__all__ = ['FocalLoss', ]


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
