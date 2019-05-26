# -*- coding: utf-8 -*-
# @Time    : 2019/5/26 12:57
# @Author  : LegenDong
# @User    : legendong
# @File    : perceptron_layer.py
# @Software: PyCharm
import math

import torch.nn as nn

__all__ = ['PerceptronLayer']


class PerceptronLayer(nn.Module):

    def __init__(self, in_features, middle_ratio=2, drop_prob=0.5, prelu_init=1, ):
        super(PerceptronLayer, self).__init__()
        self.in_features = in_features
        self.middle_ratio = middle_ratio
        self.drop_prob = drop_prob
        self.prelu_init = prelu_init

        self.fc = nn.Sequential(nn.Linear(self.in_features, math.ceil(self.in_features * self.middle_ratio)),
                                nn.BatchNorm1d(math.ceil(self.in_features * self.middle_ratio)),
                                nn.PReLU(init=self.prelu_init),
                                nn.Dropout(p=self.drop_prob),
                                nn.Linear(math.ceil(self.in_features * self.middle_ratio), self.in_features),
                                nn.BatchNorm1d(self.in_features),
                                nn.PReLU(init=self.prelu_init), )

        nn.init.kaiming_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, .0)
        nn.init.kaiming_normal_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, .0)

    def forward(self, x):
        output = self.fc(x)
        output = x + output

        return output
