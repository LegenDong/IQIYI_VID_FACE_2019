# -*- coding: utf-8 -*-
# @Time    : 2019/5/31 16:17
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_test_multi_view_maxout.py
# @Software: PyCharm
import argparse
import logging
import os
import pickle

import torch
from torch.utils.data import DataLoader

from datasets import IQiYiVidDataset
from models import ArcFaceNanMaxOutModel
from utils import check_exists, init_logging, sep_cat_qds_select_vid_transforms, get_mask_index

logger = logging.getLogger(__name__)


def main(data_root, num_frame, num_attn, moda, seed, stuff_labels, epoch):
    mask_index = get_mask_index(seed, 512, 16)
    print(mask_index)

    model_path = './checkpoints/multi_view/demo_arcface_{}_multi_view_stuff_{}_{}_model_{:0>4d}.pth' \
        .format(moda, stuff_labels, seed, epoch)
    assert check_exists(model_path)

    dataset = IQiYiVidDataset(data_root, 'test', moda, transform=sep_cat_qds_select_vid_transforms,
                              mask_index=mask_index, num_frame=num_frame)
    data_loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)

    model = ArcFaceNanMaxOutModel(len(mask_index) + 2, 10034 + 1, num_attn=num_attn, stuff_labels=stuff_labels)
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
    parser.add_argument('--data_root', default='/data/materials/', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--log_root', default='/data/logs/', type=str,
                        help='path to save log (default: /data/logs/)')
    parser.add_argument('--result_root', default='/data/result/', type=str,
                        help='path to save result (default: /data/result/)')
    parser.add_argument('--num_frame', default=40, type=int, help='size of video length (default: 40)')
    parser.add_argument('--num_attn', default=1, type=int, help='number of attention block in NAN')
    parser.add_argument('--moda', default='face', type=str, help='modal[face, head] of model train, (default: face)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")
    parser.add_argument('--seed', type=int, default=0, help="random seed for multi view (default: 0)")
    parser.add_argument('--stuff_labels', default=1000, type=int, help='stuff num for maxout model (default: 1000)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    log_root = args.log_root
    result_root = args.result_root

    result_log_path = os.path.join(log_root, 'result_log.txt')
    result_path = os.path.join(result_root, 'result.txt')
    log_path = os.path.join(log_root, 'log.txt')

    init_logging(log_path)

    all_outputs, all_video_names = main(args.data_root, args.num_frame, args.num_attn, args.moda, args.seed,
                                        args.stuff_labels, args.epoch)

    with open('./multi_view_result/output/temp_result_{}.pickle'.format(args.seed), 'wb') as fout:
        pickle.dump(all_outputs.numpy(), fout)
    with open('./multi_view_result/name/temp_names_{}.pickle'.format(args.seed), 'wb') as fout:
        pickle.dump(all_video_names, fout)

    # top100_value, top100_idxes = torch.topk(all_outputs, 100, dim=0)
    # with open(result_log_path, 'w', encoding='utf-8') as f_result_log:
    #     with open(result_path, 'w', encoding='utf-8') as f_result:
    #         for label_idx in range(1, 10034 + 1):
    #             video_names_list = ['{}.mp4'.format(all_video_names[idx]) for idx in top100_idxes[:, label_idx]]
    #             video_names_str = ' '.join(video_names_list)
    #             f_result.write('{} {}\n'.format(label_idx, video_names_str))
    #             f_result_log.write('{} {}\n'.format(label_idx, video_names_str))
