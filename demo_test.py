# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 12:41
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_test.py
# @Software: PyCharm
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiFaceDataset
from models import TestModel
from utils import check_exists, default_get_result, weighted_average_pre_progress


def main(data_root, load_path):
    assert check_exists(load_path)

    dataset = IQiYiFaceDataset(data_root, 'val', min_value=20.,
                               pre_progress=weighted_average_pre_progress, )
    data_loader = DataLoader(dataset, batch_size=20480, shuffle=True, num_workers=4)

    model = TestModel(is_train=False)

    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    all_results = []
    with torch.no_grad():
        for batch_idx, (feats, _, video_names) in enumerate(data_loader):
            feats = feats.to(device)
            output = model(feats, _)

            results = default_get_result(output.cpu(), video_names)
            all_results += list(results)

    results_dict = {}
    for result in all_results:
        key = result[0].int().item()
        if key not in results_dict.keys():
            results_dict[key] = [(*result[1:],)]
        else:
            results_dict[key].append((*result[1:],))

    with open('./results/result.txt', 'w', encoding='utf-8') as f:
        for key, value in sorted(results_dict.items(), key=lambda item: item[0]):
            value.sort(key=lambda k: k[0], reverse=True)
            value = ['{}.mp4'.format(i[1]) for i in value[:100]]
            video_names_str = ' '.join(value)
            f.write('{} {}\n'.format(key, video_names_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--root', default='/data/dcq/DataSets/iQIYI/', type=str,
                        help='path to load data (default: /data/dcq/DataSets/iQIYI/)')
    parser.add_argument('-l', '--load_path', default=None, required=True, type=str,
                        help='path to save model (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    main(args.root, args.load_path)
