# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 22:04
# @Author  : LegenDong
# @User    : legendong
# @File    : metrics.py
# @Software: PyCharm

import math

import torch
import torch.nn as nn

__all__ = ['ArcMarginProduct']

"""
reference: 
https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
"""


class ArcMarginProduct(nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        """
        Implement of large margin arc distance:

        :param s:               norm of input feature
        :param m:               margin
        :param easy_margin:
        """
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x_cosine, label):
        sine = torch.sqrt(1.0 - torch.pow(x_cosine, 2))
        phi = x_cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(x_cosine > 0, phi, x_cosine)
        else:
            phi = torch.where(x_cosine > self.th, phi, x_cosine - self.mm)

        one_hot = torch.zeros(x_cosine.size(), device=x_cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * x_cosine)
        output *= self.s

        return output
