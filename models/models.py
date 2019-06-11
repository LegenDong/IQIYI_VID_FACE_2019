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
from models.se_resnext import se_resnext50_32x4d

__all__ = ['BaseModel', 'ArcFaceModel', 'ArcFaceSEResNeXtModel', 'ArcSceneFeatModel', 'ArcFaceSceneModel', ]

SENEXT_PATH = './model_zoo/se_resnext50_32x4d-a260b3a4.pth'


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


class ArcFaceSEResNeXtModel(nn.Module):
    def __init__(self, num_classes=1000, include_top=True):
        super(ArcFaceSEResNeXtModel, self).__init__()
        self.num_classes = num_classes
        self.model_path = SENEXT_PATH
        self.include_top = include_top

        self._init_modules()

    def _init_modules(self):
        se_resnext = se_resnext50_32x4d(num_classes=1000)

        print('loading pre-trained weight for se_resnext from {}.'.format(self.model_path))
        state_dict = torch.load(self.model_path)
        se_resnext.load_state_dict({k: v for k, v in state_dict.items() if k in se_resnext.state_dict()})

        self.base_model = nn.Sequential(se_resnext.layer0, se_resnext.layer1,
                                        se_resnext.layer2, se_resnext.layer3,
                                        se_resnext.layer4, )

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.weight = Parameter(torch.FloatTensor(self.num_classes, 2048))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = self.base_model(x)
        output = self.global_avgpool(output)
        output = output.view(output.size(0), -1)

        if self.include_top:
            output = F.linear(F.normalize(output), F.normalize(self.weight))
        return output


class ArcSceneFeatModel(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(ArcSceneFeatModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Sequential(nn.Linear(self.in_features, self.in_features * 2),
                                nn.BatchNorm1d(self.in_features * 2),
                                nn.PReLU(),
                                nn.Dropout(),
                                nn.Linear(self.in_features * 2, self.in_features),
                                nn.BatchNorm1d(self.in_features),
                                nn.PReLU(),
                                )

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


class ArcFaceSceneModel(nn.Module):
    def __init__(self, face_dim, scene_dim, out_features):
        super(ArcFaceSceneModel, self).__init__()
        self.face_dim = face_dim
        self.scene_dim = scene_dim
        self.out_features = out_features

        self.mma_layer = MultiModalAttentionLayer(40)
        self.nan_layer = NanAttentionLayer(self.face_dim, 1)

        self.scene_fc = nn.Sequential(nn.Linear(self.scene_dim, self.scene_dim // 4),
                                      nn.BatchNorm1d(self.scene_dim // 4),
                                      nn.PReLU(),
                                      nn.Dropout(),
                                      nn.Linear(self.scene_dim // 4, self.scene_dim // 16),
                                      nn.BatchNorm1d(self.scene_dim // 16),
                                      nn.PReLU(), )

        nn.init.kaiming_normal_(self.scene_fc[0].weight)
        nn.init.constant_(self.scene_fc[0].bias, .0)
        nn.init.kaiming_normal_(self.scene_fc[4].weight)
        nn.init.constant_(self.scene_fc[4].bias, .0)

        self.final_fc = nn.Sequential(nn.Linear(self.face_dim + self.scene_dim // 16, self.face_dim * 2),
                                      nn.BatchNorm1d(self.face_dim * 2),
                                      nn.PReLU(),
                                      nn.Dropout(),
                                      nn.Linear(self.face_dim * 2, self.face_dim + self.scene_dim // 16),
                                      nn.BatchNorm1d(self.face_dim + self.scene_dim // 16),
                                      nn.PReLU(), )

        nn.init.kaiming_normal_(self.final_fc[0].weight)
        nn.init.constant_(self.final_fc[0].bias, .0)
        nn.init.kaiming_normal_(self.final_fc[4].weight)
        nn.init.constant_(self.final_fc[4].bias, .0)

        self.weight = Parameter(torch.FloatTensor(self.out_features, self.face_dim + self.scene_dim // 16))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feat1, feat2, ):
        feat1 = self.mma_layer(feat1)
        x_1 = self.nan_layer(feat1)
        x_2 = self.scene_fc(feat2)

        x = torch.cat([x_1, x_2], dim=-1)

        output = self.final_fc(x)
        output = x + output

        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output
