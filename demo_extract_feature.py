# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 17:10
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_extract_feature.py
# @Software: PyCharm
import argparse
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiFaceImageDataset
from models import ArcFaceSEResNetModel
from utils import load_face_from_pickle, check_exists


def main(data_root, save_dir, tvt, num_classes, batch_size):
    assert check_exists((data_root, save_dir))
    assert tvt in ['train', 'val', 'test']

    save_path = os.path.join(save_dir, 'face_{}_v2.pickle'.format(tvt))

    dataset = IQiYiFaceImageDataset(data_root, tvt='train+val-noise', is_extract=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    log_step = len(data_loader) // 100 if len(data_loader) > 100 else 1

    model = ArcFaceSEResNetModel(num_classes=num_classes, include_top=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    video_infos = {}
    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels, video_names) in enumerate(data_loader):
            images = images.to(device)
            output = model(images)
            output_np = output.cpu().numpy().astype(np.float16)
            for idx, video_name in enumerate(video_names):
                video_infos.setdefault(video_name.encode('utf-8'), []) \
                    .append(('0', (0., 0., 0., 0.), 1.0, 200., output_np[idx]))

            if batch_idx % log_step == 0 and batch_idx != 0:
                end = time.time()

                print('[{}/{} ({:.0f}%)] Time: {}'
                      .format(batch_idx * batch_size, len(dataset),
                              100.0 * batch_idx / len(data_loader), (end - start)))
                start = time.time()

    with open(save_path, 'wb') as fout:
        pickle.dump(video_infos, fout)
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IQIYI VID FACE 2019')
    parser.add_argument('--data_root', default='/data/materials', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--save_dir', default='./data/materials/feat', type=str,
                        help='path to save model (default: ./data/materials/feat)')
    parser.add_argument('--tvt', default='train', type=str,
                        help='type of the feature extract from the dataset (default: train)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=512, type=int, help='num of batch size (default: 512)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    filepath = main(args.data_root, args.save_dir, args.tvt, args.num_classes, args.batch_size)

    video_infos = load_face_from_pickle(filepath)
    assert len(video_infos) > 0
