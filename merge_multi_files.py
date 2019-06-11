# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 13:37
# @Author  : LegenDong
# @User    : legendong
# @File    : merge_multi_files.py
# @Software: PyCharm
import argparse
import logging
import os

from utils import merge_multi_view_result, init_logging

logger = logging.getLogger(__name__)

SCENE_PICKLE_ROOT = './multi_view_scene_result/'
FACE_SCENE_PICKLE_ROOT = './multi_view_face_scene_result/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--merge_type', default='face', type=str,
                        help='the pickle to merge (default: face)')
    args = parser.parse_args()

    log_root = '/data/logs/'
    log_path = os.path.join(log_root, 'log.txt')

    init_logging(log_path)

    assert args.merge_type in ['scene', 'face_scene', ]

    if args.merge_type == 'scene':
        pickle_root = SCENE_PICKLE_ROOT
    elif args.merge_type == 'face_scene':
        pickle_root = FACE_SCENE_PICKLE_ROOT
    else:
        raise RuntimeError

    output_num, all_video_names, output_sum = merge_multi_view_result(pickle_root, is_save=True)

    # all_outputs = output_sum / output_num
    #
    # result_root = '/data/result/'
    # result_path = os.path.join(result_root, 'result.txt')
    #
    # all_outputs = torch.from_numpy(all_outputs)
    # top100_value, top100_idxes = torch.topk(all_outputs, 100, dim=0)
    # with open(result_path, 'w', encoding='utf-8') as f_result:
    #     for label_idx in range(1, 10034 + 1):
    #         video_names_list = ['{}.mp4'.format(all_video_names[idx]) for idx in top100_idxes[:, label_idx]]
    #         video_names_str = ' '.join(video_names_list)
    #         f_result.write('{} {}\n'.format(label_idx, video_names_str))
