# -*- coding: utf-8 -*-
# @Time    : 2019/5/25 18:29
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_extract_scene.py
# @Software: PyCharm
import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiExtractSceneDataset
from models import ArcFaceSEResNeXtModel
from utils import check_exists, init_logging

logger = logging.getLogger(__name__)


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)
    load_path = './checkpoints/demo_arcface_fine_tune_model_{:0>4d}.pth'.format(args.epoch)

    dataset = IQiYiExtractSceneDataset(args.data_root, args.tvt, image_root='/home/dcq/img', num_frame=1)
    if len(dataset) <= 0:
        logger.error('the size of the dataset for extract scene feat cannot be {}'.format(len(dataset)))
    else:
        logger.info('the size of the dataset for extract scene feat is {}'.format(len(dataset)))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    log_step = len(data_loader) // 100 if len(data_loader) > 100 else 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('extract scene feat on {}'.format(device))

    model = ArcFaceSEResNeXtModel(args.num_classes, include_top=False)
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model = model.to(device)

    all_video_names = []
    all_image_index = []
    all_scene_feat = []

    start = time.time()
    with torch.no_grad():
        for batch_idx, (image_data, video_names, image_indexes) in enumerate(data_loader):
            image_data = image_data.to(device)

            outputs = model(image_data)

            all_video_names += list(video_names)
            all_image_index += image_indexes.tolist()
            all_scene_feat.append(outputs.cpu())

            if batch_idx % log_step == 0:
                end = time.time()
                log_info = '[{}/{} ({:.0f}%)] Time: {}' \
                    .format(batch_idx * args.batch_size, len(dataset),
                            100.0 * batch_idx / len(data_loader), (end - start))
                logger.info(log_info)
                print(log_info)
                start = time.time()

    all_scene_feat = torch.cat(all_scene_feat, dim=0).numpy()
    scene_infos = {}

    for idx, video_name in enumerate(all_video_names):
        scene_infos.setdefault(video_name, []).append((all_image_index[idx], all_scene_feat[idx]))

    with open(os.path.join(args.save_dir, 'scene_infos_{}.pickle'.format(args.tvt)), 'wb') as fout:
        pickle.dump(scene_infos, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='/data/materials', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--save_dir', default='./scene_feat/', type=str,
                        help='path to save scene feat (default: ./scene_feat/)')
    parser.add_argument('--log_root', default='/data/logs/', type=str,
                        help='path to save log (default: /data/logs/)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=512, type=int, help='bat of feature (default: 512)')
    parser.add_argument('--tvt', default='test', type=str, help='train, val or test to extract feat (default: train)')
    parser.add_argument('--epoch', default=20, type=int, help='train, val or test to extract feat (default: train)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log_root = args.log_root
    log_path = os.path.join(log_root, 'log.txt')

    if check_exists(log_path):
        os.remove(log_path)

    init_logging(log_path)

    main(args)

    with open(os.path.join(args.save_dir, 'scene_infos_{}.pickle'.format(args.tvt)), 'rb') as fin:
        scene_infos = pickle.load(fin, encoding='bytes')

    assert isinstance(scene_infos, dict)
