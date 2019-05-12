# -*- coding: utf-8 -*-
# @Time    : 2019-05-12 21:42
# @Author  : edward
# @File    : temp.py
# @Software: PyCharm
import torch
from models import NanModel

if __name__ == '__main__':
    data = torch.randn((4, 30, 512)).cuda()
    model = NanModel(512, 1000, num_attn=3, use_gpu=True).cuda()
    print(model(data).size())
