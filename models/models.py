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

from models.layer import MultiModalAttentionLayer, NanAttentionLayer

__all__ = ['BaseModel', 'ArcFaceModel', 'ArcFaceMaxOutModel', 'ArcFaceMultiModalModel', 'NanModel']


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


class ArcFaceMultiModalModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcFaceMultiModalModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Sequential(nn.Linear(self.in_features, self.in_features * 2),
                                 nn.BatchNorm1d(self.in_features * 2),
                                 nn.PReLU(),
                                 nn.Dropout(),
                                 nn.Linear(self.in_features * 2, self.in_features),
                                 nn.BatchNorm1d(self.in_features),
                                 nn.PReLU(), )

        nn.init.kaiming_normal_(self.fc1[0].weight)
        nn.init.constant_(self.fc1[0].bias, .0)
        nn.init.kaiming_normal_(self.fc1[4].weight)
        nn.init.constant_(self.fc1[4].bias, .0)

        self.fc2 = nn.Sequential(nn.Linear(self.in_features, self.in_features * 2),
                                 nn.BatchNorm1d(self.in_features * 2),
                                 nn.PReLU(),
                                 nn.Dropout(),
                                 nn.Linear(self.in_features * 2, self.in_features),
                                 nn.BatchNorm1d(self.in_features),
                                 nn.PReLU(), )

        nn.init.kaiming_normal_(self.fc2[0].weight)
        nn.init.constant_(self.fc2[0].bias, .0)
        nn.init.kaiming_normal_(self.fc2[4].weight)
        nn.init.constant_(self.fc2[4].bias, .0)

        self.multi_modal_attention_layer = MultiModalAttentionLayer(2, 4)

        self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features * 2))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x1, x2, ):
        output1 = self.fc1(x1)
        output1 = x1 + output1

        output2 = self.fc1(x2)
        output2 = x2 + output2

        output = torch.cat([output1.view(-1, 1, self.in_features),
                            output2.view(-1, 1, self.in_features)], dim=1)
        output = self.multi_modal_attention_layer(output).view(-1, self.in_features * 2)

        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class NanModel(nn.Module):
    def __init__(self, in_features, out_features, num_attn=1, use_gpu=True):
        super(NanModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_attn = num_attn
        self.use_gpu = use_gpu

        self.nan = NanAttentionLayer(self.in_features, self.num_attn, use_gpu=self.use_gpu)

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

    def forward(self, feats):
        x = self.nan(feats)

        output = self.fc(x)
        output = x + output
        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


if __name__ == '__main__':
    data = torch.randn((4, 512, 30))
    model = NanModel(512, 1000, num_attn=3, use_gpu=False)
    print(model(data).size())
