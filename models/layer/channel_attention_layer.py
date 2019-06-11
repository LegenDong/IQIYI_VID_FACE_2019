# -*- coding: utf-8 -*-
# @Time    : 2019/5/11 14:44
# @Author  : LegenDong
# @User    : legendong
# @File    : channel_attention_layer.py
# @Software: PyCharm
import torch

from torch import nn
from torch.nn import functional as F

__all__ = ['MultiModalAttentionLayer']


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class MultiModalAttentionLayer(nn.Module):
    def __init__(self, inplanes, planes=None, bn_layer=True):
        super(MultiModalAttentionLayer, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        if self.planes is None:
            self.planes = inplanes // 2

        self.W_g = conv1x1(self.inplanes, self.planes)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv1x1(self.planes, self.inplanes),
                nn.BatchNorm1d(self.inplanes)
            )
            nn.init.constant_(self.W_z[0].weight, .0)
        else:
            self.W_z = conv1x1(self.planes, self.inplanes)
            nn.init.constant_(self.W_z.weight, .0)

        if bn_layer:
            self.W_theta = nn.Sequential(
                conv1x1(self.inplanes, self.planes),
                nn.BatchNorm1d(self.planes)
            )
            nn.init.constant_(self.W_theta[0].weight, .0)

            self.W_phi = nn.Sequential(
                conv1x1(self.inplanes, self.planes),
                nn.BatchNorm1d(self.planes)
            )
            nn.init.constant_(self.W_phi[0].weight, .0)
        else:
            self.W_theta = conv1x1(self.inplanes, self.planes)
            nn.init.constant_(self.W_theta.weight, .0)

            self.W_phi = conv1x1(self.inplanes, self.planes)
            nn.init.constant_(self.W_phi.weight, .0)

    def forward(self, x):
        # print(x.size())

        # g_x -> (b, c, l) -> (b, 0.5c, l)
        g_x = self.W_g(x)
        # print(g_x.size())

        # theta_x -> (b, c, l) -> (b, 0.5c, l)
        theta_x = self.W_theta(x)
        # print(theta_x.size())

        # phi_x -> (b, c, l) -> (b, 0.5c, l) -> (b, l, 0.5c)
        phi_x = self.W_phi(x)
        phi_x = phi_x.permute(0, 2, 1)
        # print(phi_x.size())

        # f -> (b, 0.5c, l) dot (b, l, 0.5c) -> (b, 0.5c, 0.5c)
        f_x = torch.matmul(theta_x, phi_x)
        f_x = F.softmax(f_x, dim=-1)
        # print(f_x.size())

        # (b, 0.5c, 0.5c) dot (b, 0.5c, l) -> (b, 0.5c, l)
        y = torch.matmul(f_x, g_x)
        y = self.W_z(y)
        # print(y.size())

        out = y + x

        return out
