# -*- coding: utf-8 -*-
# @Time    : 2019/6/9 16:43
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_test_face_scene_norm_multi_view_only_face.py
# @Software: PyCharm
import argparse
import logging
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiFaceSceneDataset
from models import ArcFaceSceneNormModel
from utils import check_exists, init_logging, sep_cat_qds_select_face_scene_transforms

logger = logging.getLogger(__name__)


def main(face_root, scene_root, seed, epoch):
    mask_path = './checkpoints/multi_view_face_scene_norm_only_face/mask_index_file_{}.pickle'.format(seed)
    assert check_exists(mask_path)

    with open(mask_path, 'rb') as fin:
        face_mask_index = pickle.load(fin, encoding='bytes')
    print(face_mask_index)
    model_path = './checkpoints/multi_view_face_scene_norm_only_face/' \
                 'demo_arcface_face+scene_norm_only_face_{}_model_{:0>4d}.pth'.format(seed, epoch)
    assert check_exists(model_path)

    dataset = IQiYiFaceSceneDataset(face_root, scene_root, 'test', num_frame=40,
                                    transform=sep_cat_qds_select_face_scene_transforms, face_mask=face_mask_index)
    data_loader = DataLoader(dataset, batch_size=16384, shuffle=False, num_workers=0)

    model = ArcFaceSceneNormModel(len(face_mask_index) + 2, 2048, 10034 + 1, )
    metric_func = torch.nn.Softmax(-1)

    logger.info('load model from {}'.format(model_path))
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info('test model on {}'.format(device))

    model.eval()
    all_outputs = []
    all_video_names = []

    with torch.no_grad():
        for batch_idx, (feats1, feats2, _, video_names) in enumerate(data_loader):
            logger.info('Test Model: {}/{}'.format(batch_idx, len(data_loader)))

            feats1 = feats1.to(device)
            feats2 = feats2.to(device)
            output = model(feats1, feats2)
            output = metric_func(output)
            all_outputs.append(output.cpu())
            all_video_names += video_names

    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs, all_video_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--face_root', default='/data/materials', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--scene_root', default='./scene_feat', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--save_dir', default='./checkpoints/', type=str,
                        help='path to save model (default: ./checkpoints/)')
    parser.add_argument('--log_root', default='/data/logs/', type=str,
                        help='path to save log (default: /data/logs/)')
    parser.add_argument('--result_root', default='/data/result/', type=str,
                        help='path to save result (default: /data/result/)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")
    parser.add_argument('--seed', type=int, default=0, help="random seed for multi view (default: 0)")

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

    all_outputs, all_video_names = main(args.face_root, args.scene_root, args.seed, args.epoch)

    top100_value, top100_idxes = torch.topk(all_outputs, 100, dim=0)
    with open(result_log_path, 'w', encoding='utf-8') as f_result_log:
        with open(result_path, 'w', encoding='utf-8') as f_result:
            for label_idx in range(1, 10034 + 1):
                video_names_list = ['{}.mp4'.format(all_video_names[idx]) for idx in top100_idxes[:, label_idx]]
                video_names_str = ' '.join(video_names_list)
                f_result.write('{} {}\n'.format(label_idx, video_names_str))
                f_result_log.write('{} {}\n'.format(label_idx, video_names_str))
