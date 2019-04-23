# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 22:04
# @Author  : LegenDong
# @User    : legendong
# @File    : metrics.py
# @Software: PyCharm

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

__all__ = ['ArcMarginProduct']

"""
reference: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
"""


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        """
        Implement of large margin arc distance:
        :param in_features:     size of each input sample
        :param out_features:    size of each output sample
        :param s:               norm of input feature
        :param m:               margin
        :param easy_margin:
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
