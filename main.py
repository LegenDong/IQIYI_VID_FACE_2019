# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:36
# @Author  : LegenDong
# @User    : legendong
# @File    : main.py
# @Software: PyCharm
import logging
import os
import random
import time

import numpy as np
import torch

import demo_test_nan
from utils import check_exists, init_logging

logger = logging.getLogger(__name__)


def main(data_root):
    all_outputs_1, all_video_names_1 = demo_test_nan.main(data_root, 40, 1, 'face', 100)
    all_outputs_2, all_video_names_2 = demo_test_nan.main(data_root, 40, 1, 'head', 200)

    new_all_outputs = []
    new_all_video_names = []

    new_all_outputs.append(all_outputs_1)
    new_all_video_names += all_video_names_1

    for video_idx in range(all_outputs_2.size(0)):
        if all_video_names_2[video_idx] not in new_all_video_names:
            logger.info('vid {} use the result from head'.format(all_video_names_2[video_idx]))
            new_all_outputs.append(all_outputs_2[video_idx].view(1, -1))
            new_all_video_names.append(all_video_names_2[video_idx])
    new_all_outputs = torch.cat(new_all_outputs, dim=0)

    return new_all_outputs, new_all_video_names


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    SEED = int(time.time())
    logger.info('time random seed is {}'.format(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    log_root = '/data/logs/'
    result_root = '/data/result/'
    data_root = '/data/materials/'
    result_log_path = os.path.join(log_root, 'result_log.txt')
    result_path = os.path.join(result_root, 'result.txt')
    log_path = os.path.join(log_root, 'log.txt')

    if check_exists(result_log_path):
        os.remove(result_log_path)
    if check_exists(result_path):
        os.remove(result_path)
    if check_exists(log_path):
        os.remove(log_path)

    init_logging(log_path)

    all_outputs, all_video_names = main(data_root)

    top100_value, top100_idxes = torch.topk(all_outputs, 100, dim=0)
    with open(result_log_path, 'w', encoding='utf-8') as f_result_log:
        with open(result_path, 'w', encoding='utf-8') as f_result:
            for label_idx in range(1, 10034):
                video_names_list = ['{}.mp4'.format(all_video_names[idx]) for idx in top100_idxes[:, label_idx]]
                video_names_str = ' '.join(video_names_list)
                f_result.write('{} {}\n'.format(label_idx, video_names_str))
                f_result_log.write('{} {}\n'.format(label_idx, video_names_str))
