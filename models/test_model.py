# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 14:53
# @Author  : LegenDong
# @User    : legendong
# @File    : test_model.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn

from models.metrics import ArcMarginProduct, ArcMarginSplitProduct

__all__ = ['TestModel', 'TestSplitModel']


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
            pred = self.arc_margin_product(output, label)
        else:
            _x = F.normalize(output)
            _weight = F.normalize(self.arc_margin_product.weight)
            output = F.linear(_x, _weight) * self.arc_margin_product.s
            pred = self.softmax(output)

        return output, pred


class TestSplitModel(nn.Module):
    def __init__(self, is_train=True):
        super(TestSplitModel, self).__init__()
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

        self.arc_margin_product = ArcMarginSplitProduct(512, 10034 + 1,
                                                        m_arc=0.3, m_cosine=0.2)
        self.softmax = nn.Softmax(-1)

    def forward(self, x, label):
        """
        :param x: feature matrix
        :param label: ground truth
        :return:
            output: output features, can be used to calculate center loss, et al
            pred: predict results.
        """
        output = self.fc(x)
        output = x + output
        phi, cosine = self.arc_margin_product(output)
        if self.is_train:
            one_hot = torch.zeros(cosine.size(), device=x.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            pred = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            pred *= self.arc_margin_product.s
        else:
            pred = self.softmax(cosine)

        return output, pred
