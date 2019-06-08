# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:36
# @Author  : LegenDong
# @User    : legendong
# @File    : main.py
# @Software: PyCharm
import logging
import os
import pickle
import random
import time

import numpy as np
import torch

from utils import init_logging, merge_multi_view_result, split_name_by_l2norm, check_exists

logger = logging.getLogger(__name__)

FACE_TEST_NAME = 'face_test.pickle'
SPLIT_POINTS = (8.867,)
BALANCE_WEIGHT = ((0., 1.), (0.1, 0.9),)


def main():
    output_num, all_video_names, output_sum = merge_multi_view_result('./multi_view_face_result', is_save=False)
    all_outputs = torch.from_numpy(output_sum / output_num)

    split_names = split_name_by_l2norm(os.path.join('/data/materials', 'feat', FACE_TEST_NAME), SPLIT_POINTS)

    assert check_exists('./scene_result/scene_name_output_dict.pickle')
    with open('./scene_result/scene_name_output_dict.pickle', 'rb') as fin:
        name_output_dict = pickle.load(fin, encoding='bytes')

    new_all_outputs = []
    new_all_video_names = []
    for name_idx, video_name in enumerate(all_video_names):
        temp_output = all_outputs[name_idx]
        for split_idx, split in enumerate(split_names):
            if video_name in split:
                if video_name in name_output_dict:
                    balance_weight = BALANCE_WEIGHT[split_idx]
                    scene_output = torch.from_numpy(name_output_dict[video_name])
                    temp_output = temp_output * balance_weight[0] + scene_output * balance_weight[1]
                    logger.info('video {} use scene output to calc by weight ({})'
                                .format(video_name, ', '.join([str(weight) for weight in balance_weight])))
                else:
                    logger.warning('video {} should in name_output_dict but not'.format(video_name))
        new_all_outputs.append(temp_output.view(1, -1))
        new_all_video_names.append(video_name)
    for video_name in split_names[0]:
        if video_name in name_output_dict:
            balance_weight = BALANCE_WEIGHT[0]
            scene_output = torch.from_numpy(name_output_dict[video_name])
            temp_output = scene_output * balance_weight[1]
            logger.info('video {} use scene output to calc by weight ({})'
                        .format(video_name, ', '.join([str(weight) for weight in balance_weight])))
            new_all_outputs.append(temp_output.view(1, -1))
            new_all_video_names.append(video_name)
        else:
            logger.warning('video {} should in name_output_dict but not'.format(video_name))

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
    result_log_path = os.path.join(log_root, 'result_log.txt')
    result_path = os.path.join(result_root, 'result.txt')
    log_path = os.path.join(log_root, 'log.txt')

    init_logging(log_path)

    all_outputs, all_video_names = main()

    top100_value, top100_idxes = torch.topk(all_outputs, 100, dim=0)
    with open(result_log_path, 'w', encoding='utf-8') as f_result_log:
        with open(result_path, 'w', encoding='utf-8') as f_result:
            for label_idx in range(1, 10034 + 1):
                video_names_list = ['{}.mp4'.format(all_video_names[idx]) for idx in top100_idxes[:, label_idx]]
                video_names_str = ' '.join(video_names_list)
                f_result.write('{} {}\n'.format(label_idx, video_names_str))
                f_result_log.write('{} {}\n'.format(label_idx, video_names_str))
