# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:28
# @Author  : LegenDong
# @User    : legendong
# @File    : utils.py
# @Software: PyCharm
import logging
import os
import pickle
import random

import numpy as np
import torch

__all__ = ['init_logging', 'check_exists', 'load_train_gt_from_txt', 'load_val_gt_from_txt', 'load_face_from_pickle',
           'load_head_from_pickle', 'load_body_from_pickle', 'load_audio_from_pickle', 'default_get_result',
           'default_transforms', 'default_target_transforms', 'save_model', 'topk_func', 'default_face_pre_progress',
           'max_score_face_pre_progress', 'average_face_pre_progress', 'weighted_average_face_pre_progress',
           'retain_noise_in_val']

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)


def init_logging(filename, level=logging.DEBUG, log_format=LOG_FORMAT):
    logging.basicConfig(filename=filename, level=level, format=log_format)


def check_exists(file_paths):
    if not isinstance(file_paths, (list, tuple)):
        file_paths = [file_paths]

    logger.info('check_exists: check paths for {}'.format(' '.join(file_paths)))

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.info('check_exists: {} not exist'.format(file_path))
            return False
    logger.info('check_exists: {} all exist'.format(' '.join(file_paths)))
    return True


def save_model(model, save_path, name, epoch, is_best=False):
    if is_best:
        save_name = os.path.join(save_path, '{}.pth'.format(name))
        logger.info('save_model: save model in {}'.format(' '.join(save_name)))
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        save_name = os.path.join(save_path, '{}_{:0>4d}.pth'.format(name, epoch))
        logger.info('save_model: save model in {}'.format(' '.join(save_name)))
        torch.save(model.state_dict(), save_name)
    return save_name


def topk_func(output, target, k=5):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def default_get_result(output, video_names):
    values, indexes = torch.max(output, dim=1)
    return zip(indexes, values, video_names)


def max_score_face_pre_progress(video_infos, gt_infos, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        if len(frame_infos) == 0:
            continue

        frame_infos.sort(key=lambda k: (k.get('quality_score', .0)), reverse=True)
        feat = frame_infos[0]['feat']
        label = gt_infos.get(video_name, 0)

        feats.append(feat)
        labels.append(label)
        video_names.append(video_name)

    return feats, labels, video_names


def average_face_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        if len(frame_infos) == 0:
            continue

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

        feat = np.mean(np.array(temp_feats), axis=0)
        label = gt_infos.get(video_name, 0)

        feats.append(feat)
        labels.append(label)
        video_names.append(video_name)

    return feats, labels, video_names


def weighted_average_face_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        if len(frame_infos) == 0:
            continue

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

        feat = np.sum(np.array(temp_feats), axis=0) / sum_score
        label = gt_infos.get(video_name, 0)

        feats.append(feat)
        labels.append(label)
        video_names.append(video_name)

    return feats, labels, video_names


def default_face_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        if len(frame_infos) == 0:
            continue

        is_choose = False
        for frame_info in frame_infos:
            if (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                feat = frame_info['feat']
                label = gt_infos.get(video_name, 0)

                feats.append(feat)
                labels.append(label)
                video_names.append(video_name)
                is_choose = True

        if not is_choose:
            frame_info = random.choice(frame_infos)
            feat = frame_info['feat']
            label = gt_infos.get(video_name, 0)

            feats.append(feat)
            labels.append(label)

            video_names.append(video_name)

    return feats, labels, video_names


def retain_noise_in_val(feats, labels, video_names, pr=0.5, **kwargs):
    assert len(feats) == len(labels)
    assert len(labels) == len(video_names)
    assert .0 <= pr <= 1.

    idx_list = []
    for label_idx, label in enumerate(labels):
        if 'TRAIN' in video_names[label_idx] or \
                (label == 0 and 'VAL' in video_names[label_idx] and random.uniform(0, 1) < pr):
            idx_list.append(label_idx)
    feats = [feats[idx] for idx in idx_list]
    labels = [labels[idx] for idx in idx_list]
    video_names = [video_names[idx] for idx in idx_list]

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
        for line in fin.readlines():
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


def load_head_from_pickle(file_path):
    assert check_exists(file_path)

    with open(file_path, 'rb') as fin:
        head_feats_dict = pickle.load(fin, encoding='bytes')

    video_infos = []

    for video_ind, video_name in enumerate(head_feats_dict):
        head_feats = head_feats_dict[video_name]
        last_fame_num = 0
        frame_infos = []
        for ind, head_score in enumerate(head_feats):
            [frame_str, bbox, det_score, feat] = head_score
            [x1, y1, x2, y2] = bbox
            assert (int(frame_str) >= last_fame_num)
            last_fame_num = int(frame_str)
            assert (0 <= x1 <= x2)
            assert (0 <= y1 <= y2)
            assert (type(det_score) == float)
            assert (feat.dtype == np.float16 and feat.shape[0] == 512)

            frame_infos.append({'frame_id': last_fame_num,
                                'bbox': bbox,
                                'det_score': det_score,
                                'feat': feat})
        video_infos.append({
            'video_ind': video_ind,
            'video_name': video_name.decode("utf-8"),
            'frame_infos': frame_infos})

    return video_infos


def load_body_from_pickle(file_path):
    assert check_exists(file_path)

    with open(file_path, 'rb') as fin:
        body_feats_dict = pickle.load(fin, encoding='bytes')

    video_infos = []

    for video_ind, video_name in enumerate(body_feats_dict):
        body_feats = body_feats_dict[video_name]
        last_fame_num = 0
        frame_infos = []
        for ind, head_score in enumerate(body_feats):
            [frame_str, bbox, feat] = head_score
            [x1, y1, x2, y2] = bbox
            assert (int(frame_str) >= last_fame_num)
            last_fame_num = int(frame_str)
            assert (0 <= x1 <= x2)
            assert (0 <= y1 <= y2)
            assert (feat.dtype == np.float16 and feat.shape[0] == 512)

            frame_infos.append({'frame_id': last_fame_num,
                                'bbox': bbox,
                                'feat': feat})
        video_infos.append({
            'video_ind': video_ind,
            'video_name': video_name.decode("utf-8"),
            'frame_infos': frame_infos})

    return video_infos


def load_audio_from_pickle(file_path):
    assert check_exists(file_path)

    with open(file_path, 'rb') as fin:
        audio_feats_dict = pickle.load(fin, encoding='bytes')

    video_infos = []

    for video_ind, video_name in enumerate(audio_feats_dict):
        audio_feat = audio_feats_dict[video_name]
        assert (audio_feat.dtype == np.float16 and audio_feat.shape[0] == 512)

        video_infos.append({
            'video_ind': video_ind,
            'video_name': video_name.decode("utf-8"),
            'feat': audio_feat})

    return video_infos
