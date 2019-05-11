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

import demo_test_arcface
from utils import check_exists, init_logging

logger = logging.getLogger(__name__)


# def main(method_idx, data_root):
#     if method_idx == 0:
#         return demo_test_arcface.main(data_root, 'face')
#     elif method_idx == 1:
#         return demo_test_mm.main(data_root)
#     elif method_idx == 2:
#         results_1 = demo_test_arcface.main(data_root, 'face')
#         results_2 = demo_test_arcface.main(data_root, 'head')
#         results_3 = demo_test_arcface.main(data_root, 'body')
#
#         vid_names = []
#         for result in results_1:
#             vid_names.append(result[2])
#         for result in results_2:
#             if result[2] not in vid_names:
#                 results_1.append(result)
#         for result in results_3:
#             if result[2] not in vid_names:
#                 results_1.append(result)
#
#         return results_1
#     elif method_idx == 3:
#         results_1 = demo_test_mm.main(data_root)
#         results_2 = demo_test_arcface.main(data_root, 'face')
#         results_3 = demo_test_arcface.main(data_root, 'head')
#         results_4 = demo_test_arcface.main(data_root, 'body')
#
#         vid_names = []
#         for result in results_1:
#             vid_names.append(result[2])
#         for result in results_2:
#             if result[2] not in vid_names:
#                 results_1.append(result)
#         for result in results_3:
#             if result[2] not in vid_names:
#                 results_1.append(result)
#         for result in results_4:
#             if result[2] not in vid_names:
#                 results_1.append(result)
#
#         return results_1


def main(data_root):
    results_1 = demo_test_arcface.main(data_root, 'face')
    results_2 = demo_test_arcface.main(data_root, 'head')
    results_3 = demo_test_arcface.main(data_root, 'body')

    vid_names = []
    for result in results_1:
        vid_names.append(result[2])
    for result in results_2:
        if result[2] not in vid_names:
            logger.info('video {} from head model'.format(result[2]))
            results_1.append(result)
    for result in results_3:
        if result[2] not in vid_names:
            logger.info('video {} from body model'.format(result[2]))
            results_1.append(result)

    return results_1


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

    all_results = main(data_root)

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
