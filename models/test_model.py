# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 14:53
# @Author  : LegenDong
# @User    : legendong
# @File    : test_model.py
# @Software: PyCharm
import torch.nn.functional as F
from torch import nn

from models.metrics import ArcMarginProduct


class TestModel(nn.Module):
    def __init__(self, is_train=True):
        super(TestModel, self).__init__()
        self.is_train = is_train
        self.fc = nn.Sequential(nn.Linear(512, 1024),
                                nn.BatchNorm1d(1024),
                                nn.PReLU(),
                                nn.Dropout(),
                                nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.PReLU(), )

        nn.init.kaiming_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, .0)
        nn.init.kaiming_normal_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, .0)

        self.arc_margin_product = ArcMarginProduct(512, 10034 + 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, label):
        output = self.fc(x)
        output = x + output
        if self.is_train:
            output = self.arc_margin_product(output, label)
        else:
            _x = F.normalize(output)
            _weight = F.normalize(self.arc_margin_product.weight)
            output = F.linear(_x, _weight) * self.arc_margin_product.s
            output = self.softmax(output)

        return output
