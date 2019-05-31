# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 12:31
# @Author  : LegenDong
# @User    : legendong
# @File    : non_local_layer.py
# @Software: PyCharm
import torch

from torch import nn
from torch.nn import functional as F

__all__ = ['NonLocalLayer']


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class NonLocalLayer(nn.Module):
    def __init__(self, inplanes, planes=None, bn_layer=True):
        super(NonLocalLayer, self).__init__()

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
            nn.init.constant_(self.W_z[0].weight, 1e-10)
        else:
            self.W_z = conv1x1(self.planes, self.inplanes)
            nn.init.constant_(self.W_z.weight, 1e-10)

        self.W_theta = conv1x1(self.inplanes, self.planes)
        self.W_phi = conv1x1(self.inplanes, self.planes)

    def forward(self, x):
        # print(x.size())

        # g_x -> (b, c, l) -> (b, 0.5c, l) -> (b, 0.5c, l) -> (b, l, 0.5c)
        g_x = self.W_g(x)
        g_x = g_x.view(g_x.size(0), g_x.size(1), -1)
        g_x = g_x.permute(0, 2, 1)
        # print(g_x.size())

        # theta_x -> (b, c, l) -> (b, 0.5c, l) -> (b, 0.5c, l) -> (b, l, 0.5c)
        theta_x = self.W_theta(x)
        theta_x = theta_x.view(theta_x.size(0), theta_x.size(1), -1)
        theta_x = theta_x.permute(0, 2, 1)
        # print(theta_x.size())

        # phi_x -> (b, c, l) -> (b, 0.5c, l) -> (b, 0.5c, l)
        phi_x = self.W_phi(x)
        phi_x = phi_x.view(phi_x.size(0), phi_x.size(1), -1)
        # print(phi_x.size())

        # f -> (b, l, 0.5c) dot (b, 0.5c, l) -> (b, l, l)
        f_x = torch.matmul(theta_x, phi_x)
        f_x = F.softmax(f_x, dim=-1)
        # print(f_x.size())

        # (b, l, l) dot (b, l, 0.5c) -> (b, l, 0.5c)
        # -> (b, 0.5c, l) -> (b, c, l)
        y = torch.matmul(f_x, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(y.size(0), y.size(1), x.size(2), )
        y = self.W_z(y)
        # print(y.size())

        out = y + x

        return out


if __name__ == '__main__':
    img = torch.zeros(2, 4, 512)
    net = NonLocalLayer(4)
    out = net(img)
    print(out.size())
