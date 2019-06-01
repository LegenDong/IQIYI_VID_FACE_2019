# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 14:53
# @Author  : LegenDong
# @User    : legendong
# @File    : models.py
# @Software: PyCharm
import re

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from models.layer import MultiModalAttentionLayer, NanAttentionLayer, VLADLayer, PerceptronLayer
from models.se_resnext import se_resnext50_32x4d
from .densenet import densenet161
from .se_resnet import se_resnet50

__all__ = ['BaseModel', 'ArcFaceModel', 'ArcFaceMaxOutModel', 'ArcFaceMultiModalModel', 'ArcFaceNanModel',
           'ArcFaceNanMaxOutModel', 'ArcFaceVLADModel', 'ArcFaceSimpleModel', 'ArcFaceSEResNetModel',
           'ArcFaceSEResNeXtModel', 'ArcFaceSubModel', 'ArcFaceMultiModalNanModel', 'DenseNetModel',
           'ArcSceneFeatNanModel', ]

SENET_PATH = './model_zoo/senet50_vggface2.pth'
SENEXT_PATH = './model_zoo/se_resnext50_32x4d-a260b3a4.pth'
DESNET_PATH = './model_zoo/densenet161_places365.pth.tar'


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

        self.multi_modal_attention_layer = MultiModalAttentionLayer(40)
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
        feats = self.multi_modal_attention_layer(feats)
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


class ArcFaceSEResNeXtModel(nn.Module):
    def __init__(self, num_classes=1000, ):
        super(ArcFaceSEResNeXtModel, self).__init__()
        self.num_classes = num_classes
        self.model_path = SENEXT_PATH

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
        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class ArcFaceSubModel(nn.Module):
    def __init__(self, in_features, out_features, num_attn=1, middle_ratio=2., drop_prob=0.5, prelu_init=1,
                 block_num=1, include_top=True):
        super(ArcFaceSubModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_attn = num_attn
        self.middle_ratio = middle_ratio
        self.drop_prob = drop_prob
        self.prelu_init = prelu_init
        self.block_num = block_num
        self.include_top = include_top

        perceptron_list = [PerceptronLayer(in_features=self.in_features, middle_ratio=self.middle_ratio,
                                           drop_prob=self.drop_prob, prelu_init=self.prelu_init)
                           for _ in range(block_num)]

        self.nan_layer = NanAttentionLayer(self.in_features, self.num_attn)
        self.perceptron_blocks = nn.Sequential(*perceptron_list)

        self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = self.nan_layer(x)
        output = self.perceptron_blocks(output)
        if self.include_top:
            output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class ArcFaceNanMaxOutModel(nn.Module):
    def __init__(self, in_features, out_features, num_attn, stuff_labels):
        super(ArcFaceNanMaxOutModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.num_attn = num_attn
        self.object_classes = out_features - 1
        self.stuff_labels = stuff_labels

        self.multi_modal_attention_layer = MultiModalAttentionLayer(40)

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

        self.nan_layer = NanAttentionLayer(self.in_features, 1)

        self.object_weight = Parameter(torch.FloatTensor(self.object_classes, self.in_features))
        self.stuff_weight = Parameter(torch.FloatTensor(self.stuff_labels, self.in_features))
        nn.init.xavier_uniform_(self.object_weight)
        nn.init.xavier_uniform_(self.stuff_weight)

    def forward(self, feats):
        feats = self.multi_modal_attention_layer(feats)
        x = self.nan_layer(feats)
        output = self.fc(x)
        output = x + output

        stuff_output = F.linear(F.normalize(output), F.normalize(self.stuff_weight))
        maxout_output, _ = stuff_output.max(-1)
        maxout_output = maxout_output.view(-1, 1)

        objects_output = F.linear(F.normalize(output), F.normalize(self.object_weight))
        output = torch.cat([maxout_output, objects_output], dim=1)

        return output


class ArcFaceMultiModalNanModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcFaceMultiModalNanModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.nan_layer_1 = NanAttentionLayer(self.in_features, 1)
        self.nan_layer_2 = NanAttentionLayer(self.in_features, 1)

        self.balance_weight = Parameter(torch.FloatTensor([0.1]))
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, feat1, feat2, ):
        x1 = self.nan_layer_1(feat1)
        x2 = self.nan_layer_2(feat2)

        balance_weight = self.sigmoid(self.balance_weight)
        x = (x1 + x2 * balance_weight) / (1 + balance_weight)

        output = self.fc(x)
        output = x + output

        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output


class DenseNetModel(nn.Module):
    def __init__(self, num_classes=1000, include_top=False):
        super(DenseNetModel, self).__init__()
        self.num_classes = num_classes
        self.include_top = include_top

        self.model_path = DESNET_PATH

        self._init_modules()

    def _init_modules(self):
        densenet = densenet161(num_classes=1000)

        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}

        remove_data_parallel = False
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            # Delete old key only if modified.
            if match or remove_data_parallel:
                del state_dict[key]
        print('loading pre-trained weight for densenet from {}.'.format(self.model_path))
        densenet.load_state_dict({k: v for k, v in state_dict.items() if k in densenet.state_dict()})

        self.base_model = densenet.features

        self.weight = Parameter(torch.FloatTensor(self.num_classes, 2208))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        output = self.base_model(x)

        if self.include_top:
            output = F.relu(output, inplace=True)
            output = F.avg_pool2d(output, kernel_size=7, stride=1).view(output.size(0), -1)
            output = output.view(output.size(0), -1)
            output = F.linear(F.normalize(output), F.normalize(self.weight))
        else:
            output = F.avg_pool2d(output, kernel_size=7, stride=1).view(output.size(0), -1)
        return output


class ArcSceneFeatNanModel(nn.Module):
    def __init__(self, in_features, out_features, num_attn=1, num_frame=10):
        super(ArcSceneFeatNanModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_attn = num_attn
        self.num_frame = num_frame

        self.multi_modal_attention_layer = MultiModalAttentionLayer(self.num_frame)
        self.nan_layer = NanAttentionLayer(self.in_features, self.num_attn)

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

    def forward(self, feats):
        feats = self.multi_modal_attention_layer(feats)
        x = self.nan_layer(feats)

        output = self.fc(x)
        output = x + output
        output = F.linear(F.normalize(output), F.normalize(self.weight))

        return output
