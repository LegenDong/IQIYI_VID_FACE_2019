# -*- coding: utf-8 -*-
# @Time    : 2019-05-11 13:51
# @Author  : edward
# @File    : demo_test_nan.py
# @Software: PyCharm

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiSepDataset
from models import NanModel
from utils import check_exists, default_get_result, init_logging, makedir

logger = logging.getLogger(__name__)


def main(args):
    makedir(args.log_root)
    makedir(args.result_root)

    result_log_path = os.path.join(args.log_root, 'result_log.txt')
    result_path = os.path.join(args.result_root, 'result.txt')
    log_path = os.path.join(args.log_root, 'log.txt')

    if check_exists(result_log_path):
        os.remove(result_log_path)
    if check_exists(result_path):
        os.remove(result_path)
    if check_exists(log_path):
        os.remove(log_path)

    init_logging(log_path)

    assert check_exists(args.load_path)

    dataset = IQiYiSepDataset(args.data_root, 'test', embedding_size=args.embedding_size)
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=4)

    model = NanModel(args.feat_dim, args.num_classes, num_attn=args.num_attn)
    metric_func = torch.nn.Softmax(-1)

    logger.info('load model from {}'.format(args.load_path))
    state_dict = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info('test model on {}'.format(device))

    model.eval()
    all_results = []

    with torch.no_grad():
        for batch_idx, (feats, _, _, _, video_names) in enumerate(data_loader):
            logger.info('Test Model: {}/{}'.format(batch_idx, len(data_loader)))

            feats = feats.to(device)
            output = model(feats)
            output = metric_func(output)

            results = default_get_result(output.cpu(), video_names)
            all_results += list(results)

    results_dict = {}
    for result in all_results:
        key = result[0].int().item()
        if key not in results_dict.keys():
            results_dict[key] = [(*result[1:],)]
        else:
            results_dict[key].append((*result[1:],))

    with open(result_path, 'w', encoding='utf-8') as f:
        for key, value in sorted(results_dict.items(), key=lambda item: item[0]):
            if key > 0:
                value.sort(key=lambda k: k[0], reverse=True)
                value = ['{}.mp4'.format(i[1]) for i in value[:100]]
                video_names_str = ' '.join(value)
                f.write('{} {}\n'.format(key, video_names_str))

    with open(result_log_path, 'w', encoding='utf-8') as f:
        for key, value in sorted(results_dict.items(), key=lambda item: item[0]):
            if key > 0:
                value.sort(key=lambda k: k[0], reverse=True)
                value = ['{}.mp4'.format(i[1]) for i in value[:100]]
                video_names_str = ' '.join(value)
                f.write('{} {}\n'.format(key, video_names_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='/data/materials/', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--load_path', default=None, required=True, type=str,
                        help='path to save model (default: None)')
    parser.add_argument('--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--log_root', default='./logs/', type=str,
                        help='path to save log (default: ./logs/)')
    parser.add_argument('--result_root', default='./result/', type=str,
                        help='path to save result (default: ./result/)')
    parser.add_argument('-dim', '--feat_dim', default=512, type=int,
                        help='dim of feature (default: 512)')
    parser.add_argument('-n', '--num_classes', default=10035, type=int,
                        help='number of classes (default: 10035)')
    parser.add_argument('-e', '--embedding_size', default=30, type=int,
                        help='size of video length (default: 479)')
    parser.add_argument('-attn', '--num_attn', default=1, type=int,
                        help='number of attention block in NAN')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    main(args)
