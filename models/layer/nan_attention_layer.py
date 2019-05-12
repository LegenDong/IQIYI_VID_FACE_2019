# -*- coding: utf-8 -*-
# @Time    : 2019-05-12 21:14
# @Author  : edward
# @File    : nan_attention_layer.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NanAttentionLayer']


class NanAttentionLayer(nn.Module):
    """
        using NAN Module to aggregate features into single feature
        reference: Neural Aggregation Network for Video Face Recognition, CVPR 2017
    """

    def __init__(self, feat_dim, num_attn=1, use_gpu=True):
        super(NanAttentionLayer, self).__init__()
        self.feat_dim = feat_dim
        self.num_attn = num_attn

        self.tanh = nn.Tanh()
        self.q = nn.Parameter(torch.ones((1, 1, self.feat_dim)) * .0)

        if use_gpu:
            self.attns = [NanAttentionBlock().cuda() for i in range(self.num_attn)]
            self.fcs = [nn.Linear(self.feat_dim, self.feat_dim).cuda() for i in range(self.num_attn - 1)]
        else:
            self.attns = [NanAttentionBlock() for i in range(self.num_attn)]
            self.fcs = [nn.Linear(self.feat_dim, self.feat_dim) for i in range(self.num_attn - 1)]

        for fc in self.fcs:
            fc.weight.data.zero_()
            fc.bias.data.zero_()

    def forward(self, xs):
        """
        :param xs: feature embedding with size B*E*D where E is embedding_size/frame_number
        :return: single aggregated feature with size B*D
        """
        B, E, D = xs.shape
        xs = xs.transpose(1, 2)

        r = self.attns[0](self.q, xs)

        for i in range(self.num_attn - 1):
            q = self.tanh(self.fcs[i](r)).view(B, 1, D)
            r = self.attns[i + 1](q, xs)

        return r


class NanAttentionBlock(nn.Module):
    """
    no learnable parameters, just a callable block.
    """
    def __init__(self):
        super(NanAttentionBlock, self).__init__()

    def forward(self, q, xs):
        """
        :param q: B*1*D, parameter in aggregation module
        :param xs: B*D*E
        :return: B*D
        """
        e = torch.matmul(q, xs)
        e = F.softmax(e, dim=-1)

        r = torch.mul(xs, e)
        r = torch.sum(r, dim=-1)

        return r
