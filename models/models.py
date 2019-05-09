# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 14:53
# @Author  : LegenDong
# @User    : legendong
# @File    : models.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

__all__ = ['BaseModel', 'ArcFaceModel', 'ArcFaceMaxOutModel']


class BaseModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(BaseModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.in_features * 2),
                                nn.BatchNorm1d(self.in_features * 2),
                                nn.PReLU(),
                                nn.Dropout(),
                                nn.Linear(self.in_features * 2, self.in_features),
                                nn.BatchNorm1d(self.in_features),
                                nn.PReLU(),
                                nn.Linear(self.in_features, self.out_features))

        nn.init.kaiming_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, .0)
        nn.init.kaiming_normal_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, .0)

    def forward(self, x):
        output = self.fc(x)
        return output


class ArcFaceModel(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(ArcFaceModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.in_features * 2),
                                nn.BatchNorm1d(self.in_features * 2),
                                nn.PReLU(),
                                nn.Dropout(),
                                nn.Linear(self.in_features * 2, self.in_features),
                                nn.BatchNorm1d(self.in_features),
                                nn.PReLU(), )

        nn.init.kaiming_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, .0)
        nn.init.kaiming_normal_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, .0)

        self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = self.fc(x)
        output = x + output
        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class ArcFaceMaxOutModel(nn.Module):
    def __init__(self, in_features, out_features, stuff_labels):
        super(ArcFaceMaxOutModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.object_classes = out_features - 1
        self.stuff_labels = stuff_labels

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.in_features * 2),
                                nn.BatchNorm1d(self.in_features * 2),
                                nn.PReLU(),
                                nn.Dropout(),
                                nn.Linear(self.in_features * 2, self.in_features),
                                nn.BatchNorm1d(self.in_features),
                                nn.PReLU(), )

        nn.init.kaiming_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, .0)
        nn.init.kaiming_normal_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, .0)

        self.object_weight = Parameter(torch.FloatTensor(self.object_classes, self.in_features))
        self.stuff_weight = Parameter(torch.FloatTensor(self.stuff_labels, self.in_features))
        nn.init.xavier_uniform_(self.object_weight)
        nn.init.xavier_uniform_(self.stuff_weight)

    def forward(self, x):
        output = self.fc(x)
        output = x + output

        stuff_output = F.linear(F.normalize(output), F.normalize(self.stuff_weight))
        maxout_output, _ = stuff_output.max(-1)
        maxout_output = maxout_output.view(-1, 1)

        objects_output = F.linear(F.normalize(output), F.normalize(self.object_weight))
        output = torch.cat([maxout_output, objects_output], dim=1)

        return output


if __name__ == '__main__':
    data = torch.randn((128, 512))
    model = ArcFaceMaxOutModel(512, 1000, 200)
    print(model(data).size())
