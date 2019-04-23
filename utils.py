# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:28
# @Author  : LegenDong
# @User    : legendong
# @File    : utils.py
# @Software: PyCharm
import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch

__all__ = ['check_exists', 'load_train_gt_from_txt', 'load_val_gt_from_txt',
           'load_face_from_pickle', 'default_pre_progress', 'default_transforms',
           'default_target_transforms', 'save_model', 'max_score_pre_progress',
           'default_get_result', 'average_pre_progress', 'weighted_average_pre_progress']


def check_exists(file_paths):
    if not isinstance(file_paths, (list, tuple)):
        file_paths = [file_paths]
    for file_path in file_paths:
        if not os.path.exists(file_path):
            return False
    return True


def save_model(model, save_path, name, epoch):
    save_time = datetime.now().strftime('%m%d_%H%M%S')
    save_name = os.path.join(save_path,
                             '{}_{:0>4d}_{}.pth'.format(name, epoch, save_time))
    torch.save(model.state_dict(), save_name)
    return save_name


def default_get_result(output, video_names):
    values, indexes = torch.max(output, dim=1)
    return zip(indexes, values, video_names)


def max_score_pre_progress(video_infos, gt_infos, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']
        frame_infos.sort(key=lambda k: (k.get('quality_score', .0)), reverse=True)
        feats.append(frame_infos[0]['feat'])
        labels.append(gt_infos.get(video_name, -1))
        video_names.append(video_name)

    return feats, labels, video_names


def average_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        temp_feats = []
        max_score = -1.0
        max_feat = None
        for frame_info in frame_infos:
            if frame_info['quality_score'] > max_score:
                max_feat = frame_info['feat']
                max_score = frame_info['quality_score']

            if (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                temp_feats.append(frame_info['feat'])

        if len(temp_feats) == 0 and max_feat is not None:
            temp_feats.append(max_feat)

        if len(temp_feats) == 0:
            continue

        feats.append(np.mean(np.array(temp_feats), axis=0))
        labels.append(gt_infos.get(video_name, -1))
        video_names.append(video_name)

    return feats, labels, video_names


def weighted_average_pre_progress(video_infos, gt_infos, max_value=None, min_value=None,
                                  **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        temp_feats = []
        sum_score = .0
        max_score = -1.0
        max_feat = None
        for frame_info in frame_infos:
            if frame_info['quality_score'] > max_score:
                max_feat = frame_info['feat']
                max_score = frame_info['quality_score']

            if (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                temp_feats.append(frame_info['feat'] * frame_info['quality_score'] / 100.)
                sum_score += frame_info['quality_score'] / 100.

        if len(temp_feats) == 0 and max_feat is not None:
            temp_feats.append(max_feat * max_score / 100.)
            sum_score = max_score / 100.

        if len(temp_feats) == 0:
            continue

        feats.append(np.sum(np.array(temp_feats), axis=0) / sum_score)
        labels.append(gt_infos.get(video_name, -1))
        video_names.append(video_name)

    return feats, labels, video_names


def default_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        is_choose = False
        for frame_info in frame_infos:
            if (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                feats.append(frame_info['feat'])
                labels.append(gt_infos.get(video_name, -1))
                video_names.append(video_name)
                is_choose = True

        if not is_choose:
            frame_info = random.choice(frame_infos)
            feats.append(frame_info['feat'])
            labels.append(gt_infos.get(video_name, -1))
            video_names.append(video_name)

    return feats, labels, video_names


def default_transforms(feat, **kwargs):
    feat_np = np.array(feat)
    feat_torch = torch.from_numpy(feat_np).float()
    return feat_torch


def default_target_transforms(label, **kwargs):
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def load_train_gt_from_txt(file_path):
    assert check_exists(file_path)

    train_gt_infos = {}
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            video_name, label = line.strip().split(' ')
            train_gt_infos[video_name.replace('.mp4', '')] = int(label)

    return train_gt_infos


def load_val_gt_from_txt(file_path):
    if file_path is None:
        return {}
    assert check_exists(file_path)

    val_gt_infos = {}
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            splits = line.strip().split(' ')
            for i in range(1, len(splits)):
                val_gt_infos[splits[i].replace('.mp4', '')] = int(splits[0])

    return val_gt_infos


def load_face_from_pickle(file_path):
    assert check_exists(file_path)

    with open(file_path, 'rb') as fin:
        face_feats_dict = pickle.load(fin, encoding='bytes')

    video_infos = []

    for video_ind, video_name in enumerate(face_feats_dict):
        face_feats = face_feats_dict[video_name]
        last_fame_num = 0
        frame_infos = []
        for ind, face_feat in enumerate(face_feats):
            [frame_str, bbox, det_score, quality_score, feat] = face_feat
            [x1, y1, x2, y2] = bbox
            assert (int(frame_str) >= last_fame_num)
            last_fame_num = int(frame_str)
            assert (0 <= x1 <= x2)
            assert (0 <= y1 <= y2)
            assert (type(det_score) == float)
            assert (type(quality_score) == float)
            assert (feat.dtype == np.float16 and feat.shape[0] == 512)

            frame_infos.append({'frame_id': last_fame_num,
                                'bbox': bbox,
                                'det_score': det_score,
                                'quality_score': quality_score,
                                'feat': feat})
        video_infos.append({
            'video_ind': video_ind,
            'video_name': video_name.decode("utf-8"),
            'frame_infos': frame_infos})

    return video_infos
