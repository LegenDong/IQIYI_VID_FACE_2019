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

from models.layer import MultiModalAttentionLayer, NanAttentionLayer, VLADLayer
from .se_resnet import se_resnet50

__all__ = ['BaseModel', 'ArcFaceModel', 'ArcFaceMaxOutModel', 'ArcFaceMultiModalModel', 'ArcFaceNanModel',
           'ArcFaceVLADModel', 'ArcFaceSimpleModel', 'ArcFaceSEResNetModel']

SENET_PATH = './model_zoo/senet50_vggface2.pth'


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


class ArcFaceSimpleModel(nn.Module):
    def __init__(self, in_features, out_features, ):
        super(ArcFaceSimpleModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = F.linear(F.normalize(x), F.normalize(self.weight))

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


class ArcFaceNanModel(nn.Module):
    def __init__(self, in_features, out_features, num_attn=1):
        super(ArcFaceNanModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_attn = num_attn

        self.nan_layer = NanAttentionLayer(self.in_features, self.num_attn)

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
        x = self.nan_layer(feats)

        output = self.fc(x)
        output = x + output
        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class ArcFaceVLADModel(nn.Module):
    def __init__(self, in_features, out_features, num_cluster=3):
        super(ArcFaceVLADModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_cluster = num_cluster

        self.middle_features = in_features * num_cluster

        self.vlad_layer = VLADLayer(self.in_features, self.num_cluster)

        self.fc = nn.Sequential(nn.Linear(self.middle_features, self.middle_features * 2),
                                nn.BatchNorm1d(self.middle_features * 2),
                                nn.PReLU(),
                                nn.Dropout(),
                                nn.Linear(self.middle_features * 2, self.in_features),
                                nn.BatchNorm1d(self.in_features),
                                nn.PReLU(), )

        nn.init.kaiming_normal_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, .0)
        nn.init.kaiming_normal_(self.fc[4].weight)
        nn.init.constant_(self.fc[4].bias, .0)

        self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = x.transpose(1, 2).unsqueeze(-1)
        output = self.vlad_layer(output)

        output = self.fc(output) + output
        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class ArcFaceSEResNetModel(nn.Module):
    def __init__(self, num_classes=1000, include_top=True):
        super(ArcFaceSEResNetModel, self).__init__()
        self.num_classes = num_classes
        self.include_top = include_top

        self.model_path = SENET_PATH

        self._init_modules()

    def _init_modules(self):
        se_resnet = se_resnet50(num_classes=8631)

        print('loading pre-trained weight for se_resnet from {}.'.format(self.model_path))
        state_dict = torch.load(self.model_path)
        se_resnet.load_state_dict({k: v for k, v in state_dict.items() if k in se_resnet.state_dict()})

        self.base_model = nn.Sequential(se_resnet.conv1, se_resnet.bn1, se_resnet.relu, se_resnet.maxpool,
                                        se_resnet.layer1, se_resnet.layer2, se_resnet.layer3, se_resnet.layer4)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.weight = Parameter(torch.FloatTensor(self.num_classes, 512 * se_resnet.layer4[-1].expansion))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = self.base_model(x)
        output = self.global_avgpool(output)
        output = output.view(output.size(0), -1)

        if self.include_top:
            output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output
