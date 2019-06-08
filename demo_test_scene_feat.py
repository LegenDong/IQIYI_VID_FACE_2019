# -*- coding: utf-8 -*-
# @Time    : 2019/6/1 14:22
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_test_scene_feat.py
# @Software: PyCharm
import argparse
import logging
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiSceneFeatDataset
from models import ArcSceneFeatModel
from utils import check_exists, init_logging

logger = logging.getLogger(__name__)


def main(data_root, epoch):
    load_path = './checkpoints/demo_arcface_scene_model_{:0>4d}.pth'.format(epoch)
    assert check_exists(load_path)

    dataset = IQiYiSceneFeatDataset(data_root, 'test', )
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)

    model = ArcSceneFeatModel(2048, 10034 + 1)
    metric_func = torch.nn.Softmax(-1)

    logger.info('load model from {}'.format(load_path))
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info('test model on {}'.format(device))

    model.eval()

    all_outputs = []
    all_video_names = []
    with torch.no_grad():
        for batch_idx, (feats, _, video_names) in enumerate(data_loader):
            logger.info('Test Model: {}/{}'.format(batch_idx, len(data_loader)))

            feats = feats.to(device)
            output = model(feats)
            output = metric_func(output)
            all_outputs.append(output.cpu())
            all_video_names += video_names

    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs, all_video_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='./scene_feat/', type=str,
                        help='path to load data (default: ./scene_feat/)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--log_root', default='/data/logs/', type=str,
                        help='path to save log (default: /data/logs/)')
    parser.add_argument('--result_root', default='/data/result/', type=str,
                        help='path to save result (default: /data/result/)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")

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

    result_log_path = os.path.join(log_root, 'result_log.txt')
    result_path = os.path.join(result_root, 'result.txt')
    log_path = os.path.join(log_root, 'log.txt')

    init_logging(log_path)

    all_outputs, all_video_names = main(args.data_root, args.epoch)
    all_outputs_np = all_outputs.numpy()

    name_output_dict = {}
    for name_idx, video_name in enumerate(all_video_names):
        name_output_dict[video_name] = all_outputs_np[name_idx]

    with open('./scene_result/scene_name_output_dict.pickle', 'wb') as fout:
        pickle.dump(name_output_dict, fout)

    # top100_value, top100_idxes = torch.topk(all_outputs, 100, dim=0)
    # with open(result_log_path, 'w', encoding='utf-8') as f_result_log:
    #     with open(result_path, 'w', encoding='utf-8') as f_result:
    #         for label_idx in range(1, 10034 + 1):
    #             video_names_list = ['{}.mp4'.format(all_video_names[idx]) for idx in top100_idxes[:, label_idx]]
    #             video_names_str = ' '.join(video_names_list)
    #             f_result.write('{} {}\n'.format(label_idx, video_names_str))
    #             f_result_log.write('{} {}\n'.format(label_idx, video_names_str))
