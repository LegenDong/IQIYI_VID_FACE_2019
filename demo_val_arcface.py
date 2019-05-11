# -*- coding: utf-8 -*-
# @Time    : 2019/4/22 12:41
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_val_arcface.py
# @Software: PyCharm
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiFaceDataset, IQiYiHeadDataset, IQiYiBodyDataset
from models import ArcFaceModel
from utils import check_exists, default_get_result, weighted_average_face_pre_progress, init_logging, \
    average_pre_progress

logger = logging.getLogger(__name__)


def main(data_root, moda, ):
    load_path = './checkpoints/demo_arcface_{}_model_0100.pth'.format(moda)
    assert check_exists(load_path)

    if moda == 'face':
        dataset = IQiYiFaceDataset(data_root, 'val', pre_progress=weighted_average_face_pre_progress)
    elif moda == 'head':
        dataset = IQiYiHeadDataset(data_root, 'val', pre_progress=average_pre_progress)
    elif moda == 'body':
        dataset = IQiYiBodyDataset(data_root, 'val', pre_progress=average_pre_progress)
    else:
        raise RuntimeError

    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=4)

    model = ArcFaceModel(512, 10034 + 1)
    metric_func = torch.nn.Softmax(-1)

    logger.info('load model from {}'.format(load_path))
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info('val model on {}'.format(device))

    model.eval()
    all_results = []

    with torch.no_grad():
        for batch_idx, (feats, _, video_names) in enumerate(data_loader):
            logger.info('Val Model: {}/{}'.format(batch_idx, len(data_loader)))

            feats = feats.to(device)
            output = model(feats)
            output = metric_func(output)

            results = default_get_result(output.cpu(), video_names)
            all_results += list(results)

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='/data/materials/', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--moda', default='face', type=str,
                        help='the modal[face, head, body] use for train (default: face)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--log_root', default='/data/logs/', type=str,
                        help='path to save log (default: /data/logs/)')
    parser.add_argument('--result_root', default='/data/result/', type=str,
                        help='path to save result (default: /data/result/)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    log_root = args.log_root
    result_root = args.result_root

    result_log_path = os.path.join(log_root, 'val_result_log.txt')
    result_path = os.path.join(result_root, 'val_result.txt')
    log_path = os.path.join(log_root, 'val_log.txt')

    if check_exists(result_log_path):
        os.remove(result_log_path)
    if check_exists(result_path):
        os.remove(result_path)
    if check_exists(log_path):
        os.remove(log_path)

    init_logging(log_path)

    all_results = main(args.data_root, args.moda, )

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
