# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 19:25
# @Author  : LegenDong
# @User    : legendong
# @File    : vlad_layer.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['VLADLayer']


class VLADLayer(nn.Module):
    def __init__(self, dim=512, num_clusters=3, ghost=1, alpha=100.0, is_normalize=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            is_normalize : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(VLADLayer, self).__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.ghost = ghost
        self.alpha = alpha
        self.normalize_input = is_normalize
        self.conv = nn.Conv2d(dim, num_clusters + self.ghost, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters + self.ghost, dim))
        self._init_params()

    def _init_params(self):
        nn.init.orthogonal_(self.centroids.data)

        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.size()[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters + self.ghost, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters + self.ghost, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        residual = residual[:, :-self.ghost, :, :]
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


if __name__ == '__main__':
    data = torch.rand(256, 512, 7, 1)
    model = VLADLayer()
    output = model(data)
    print(output.size())
