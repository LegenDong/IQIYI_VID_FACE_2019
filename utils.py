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
           'default_transforms', 'default_target_transforms', 'save_model', 'topk_func', 'default_pre_progress',
           'max_score_face_pre_progress', 'average_pre_progress', 'weighted_average_face_pre_progress',
           'default_retain_noise_in_val', 'default_vid_pre_progress', 'default_vid_retain_noise_in_val',
           'default_vid_transforms', 'default_vid_target_transforms', 'default_vid_remove_noise_in_val',
           'default_remove_noise_in_val', 'sep_vid_transforms', 'sep_cat_qds_vid_transforms', 'sep_identity_transforms',
           'default_identity_target_transforms', 'default_identity_transforms', 'default_identity_pre_progress',
           'default_image_pre_progress', 'default_image_transforms', 'default_image_target_transforms',
           'prepare_device']

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)


def init_logging(filename, level=logging.DEBUG, log_format=LOG_FORMAT):
    logging.basicConfig(filename=filename, level=level, format=log_format)


def prepare_device():
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    list_ids = list(range(n_gpu))
    return device, list_ids


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
        torch.save(model.state_dict(), save_name)
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


def weighted_average_face_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, use_random=True,
                                       **kwargs):
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
        for frame_info in frame_infos:
            if (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                temp_feats.append(frame_info['feat'] * frame_info['quality_score'] / 200.)
                sum_score += frame_info['quality_score'] / 200.

        if len(temp_feats) == 0:
            if use_random:
                frame_info = random.choice(frame_infos)
                temp_feats.append(frame_info['feat'] * frame_info['quality_score'] / 200.)
                sum_score += frame_info['quality_score'] / 200.
            else:
                continue

        feat = np.sum(np.array(temp_feats), axis=0) / sum_score
        label = gt_infos.get(video_name, 0)

        feats.append(feat)
        labels.append(label)
        video_names.append(video_name)

    return feats, labels, video_names


def average_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, is_face=True, use_random=True,
                         **kwargs):
    feats = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        if len(frame_infos) == 0:
            continue

        temp_feats = []
        for frame_info in frame_infos:
            if (not is_face) or (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                temp_feats.append(frame_info['feat'])

        if len(temp_feats) == 0:
            if use_random:
                frame_info = random.choice(frame_infos)
                temp_feats.append(frame_info['feat'])
            else:
                continue

        feat = np.mean(np.array(temp_feats), axis=0)
        label = gt_infos.get(video_name, 0)

        feats.append(feat)
        labels.append(label)
        video_names.append(video_name)

    return feats, labels, video_names


def default_pre_progress(video_infos, gt_infos, max_value=None, min_value=None, is_face=True, use_random=True,
                         **kwargs):
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
            if (not is_face) or (max_value is None or frame_info['quality_score'] < max_value) \
                    and (min_value is None or frame_info['quality_score'] > min_value):
                feat = frame_info['feat']
                label = gt_infos.get(video_name, 0)

                feats.append(feat)
                labels.append(label)
                video_names.append(video_name)
                is_choose = True

        if not is_choose and use_random:
            frame_info = random.choice(frame_infos)
            feat = frame_info['feat']
            label = gt_infos.get(video_name, 0)

            feats.append(feat)
            labels.append(label)

            video_names.append(video_name)

    return feats, labels, video_names


def default_image_pre_progress(video_infos, gt_infos, image_root, **kwargs):
    file_paths = []
    bboxes = []
    labels = []
    video_names = []
    for video_info in video_infos:
        video_name = video_info['video_name']
        frame_infos = video_info['frame_infos']

        if len(frame_infos) == 0:
            continue

        for frame_info in frame_infos:
            bbox = frame_info['bbox']
            label = gt_infos.get(video_name, 0)
            frame_id = frame_info['frame_id']
            file_path = os.path.join(image_root, video_name, '{:0>6d}.jpg'.format(frame_id))

            bboxes.append(bbox)
            file_paths.append(file_path)
            labels.append(label)
            video_names.append(video_name)
    return file_paths, labels, video_names, bboxes


def default_vid_pre_progress(video_infos, gt_infos, **kwargs):
    vid_infos = {}
    for key, values in video_infos.items():
        for value in values:
            frame_infos = value['frame_infos']
            if len(frame_infos) > 0:
                vid_infos.setdefault(value['video_name'], {})[key] = frame_infos
    to_dels = []
    for key, value in vid_infos.items():
        if len(value.keys()) == len(video_infos.keys()):
            vid_infos[key]['label'] = gt_infos.get(key, 0)
            vid_infos[key]['video_name'] = key
        elif len(value.keys()) < len(video_infos.keys()):
            to_dels.append(key)
        elif len(value.keys()) > len(video_infos.keys()):
            logger.error('the vid {} has wrong num of the moda {}'.format(key, ' '.join(value.keys())))
    for to_del in to_dels:
        del vid_infos[to_del]
    return list(vid_infos.values())


def default_identity_pre_progress(video_infos, gt_infos, pr=1., **kwargs):
    vid_infos = {}
    for key, values in video_infos.items():
        for value in values:
            frame_infos = value['frame_infos']
            video_name = value['video_name']
            if len(frame_infos) > 0 and \
                    (('TRAIN' in video_name) or
                     (gt_infos.get(key, 0) != 0 and 'VAL' in video_name and random.uniform(0, 1) < pr)):
                vid_infos.setdefault(gt_infos.get(value['video_name'], 0), {}).setdefault(key, []).extend(frame_infos)
    to_dels = []
    for key, value in vid_infos.items():
        if len(value.keys()) == len(video_infos.keys()):
            vid_infos[key]['label'] = key
        elif len(value.keys()) < len(video_infos.keys()):
            to_dels.append(key)
        elif len(value.keys()) > len(video_infos.keys()):
            logger.error('the vid {} has wrong num of the moda {}'.format(key, ' '.join(value.keys())))
    for to_del in to_dels:
        del vid_infos[to_del]
    return list(vid_infos.values())


def default_vid_retain_noise_in_val(vid_infos, pr=0.5, **kwargs):
    idx_list = []
    for idx, vid_info in enumerate(vid_infos):
        if ('TRAIN' in vid_info['video_name']) or \
                (vid_info['label'] == 0 and 'VAL' in vid_info['video_name'] and random.uniform(0, 1) < pr):
            idx_list.append(idx)
    return [vid_infos[idx] for idx in idx_list]


def default_vid_remove_noise_in_val(vid_infos, **kwargs):
    idx_list = []
    for idx, vid_info in enumerate(vid_infos):
        if ('TRAIN' in vid_info['video_name']) or (vid_info['label'] != 0 and 'VAL' in vid_info['video_name']):
            idx_list.append(idx)
    return [vid_infos[idx] for idx in idx_list]


def default_retain_noise_in_val(feats, labels, video_names, pr=0.5, **kwargs):
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


def default_remove_noise_in_val(feats, labels, video_names, **kwargs):
    assert len(feats) == len(labels)
    assert len(labels) == len(video_names)

    idx_list = []
    for label_idx, label in enumerate(labels):
        if 'TRAIN' in video_names[label_idx] or (label != 0 and 'VAL' in video_names[label_idx]):
            idx_list.append(label_idx)
    feats = [feats[idx] for idx in idx_list]
    labels = [labels[idx] for idx in idx_list]
    video_names = [video_names[idx] for idx in idx_list]

    return feats, labels, video_names


def default_vid_transforms(vid_info, modes, num_frame=15, **kwargs):
    result = []
    for mode in modes:
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frames_info in frames_infos:
            temp_feats.append(frames_info['feat'])
        mean_feat = np.mean(np.array(temp_feats), axis=0)
        result.append(torch.from_numpy(mean_feat).float())
    return result


def sep_vid_transforms(vid_info, modes, num_frame=15, **kwargs):
    result = []
    for mode in modes:
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frames_info in frames_infos:
            temp_feats.append(frames_info['feat'])
        feats = np.array(temp_feats)
        result.append(torch.from_numpy(feats).float())
    return result


def sep_cat_qds_vid_transforms(vid_info, modes, num_frame=15, norm_value=100., **kwargs):
    result = []
    for mode in modes:
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frame_info in frames_infos:
            feat = frame_info['feat']
            if mode == 'face':
                feat = np.append(feat, frame_info['quality_score'] / norm_value)
            else:
                feat = np.append(feat, frame_info['det_score'])
            feat = np.append(feat, frame_info['det_score'])
            temp_feats.append(feat)
        feats = np.array(temp_feats)
        result.append(torch.from_numpy(feats).float())
    return result


def sep_identity_transforms(vid_info, modes, num_frame=15, **kwargs):
    result = []
    for mode in modes:
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frames_info in frames_infos:
            temp_feats.append(frames_info['feat'])
        feats = np.array(temp_feats)
        result.append(torch.from_numpy(feats).float())
    return result


def default_identity_transforms(vid_info, modes, num_frame=15, **kwargs):
    result = []
    for mode in modes:
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frames_info in frames_infos:
            temp_feats.append(frames_info['feat'])
        mean_feat = np.mean(np.array(temp_feats), axis=0)
        result.append(torch.from_numpy(mean_feat).float())
    return result


def default_image_transforms(image_data, bbox, mean_bgr, augm_func, ratio, **kwargs):
    bbox = _padding_bbox(bbox, ratio)
    image_data = image_data.crop(bbox)
    image_data = augm_func(image_data)
    image_data = np.array(image_data, dtype=np.uint8)
    image_data = _trans_img(image_data, mean_bgr)

    return image_data


def _trans_img(img, mean_bgr):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img


def _padding_bbox(bbox, ratio=1.):
    x1, y1, x2, y2 = bbox
    x_padding = (ratio - 1) * (x2 - x1) / 2
    y_padding = (ratio - 1) * (y2 - y1) / 2
    bbox = (x1 - x_padding, y1 - y_padding, x2 + x_padding, y2 + y_padding)

    return bbox


def default_transforms(feat, **kwargs):
    feat_np = np.array(feat)
    feat_torch = torch.from_numpy(feat_np).float()
    return feat_torch


def default_vid_target_transforms(vid_info, **kwargs):
    label = vid_info['label']
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def default_identity_target_transforms(vid_info, **kwargs):
    label = vid_info['label']
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def default_target_transforms(label, **kwargs):
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def default_image_target_transforms(label, **kwargs):
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
            assert (feat.dtype == np.float16 and (feat.shape[0] == 512 or feat.shape[0] == 2048))

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
        for ind, body_feat in enumerate(body_feats):
            [frame_str, bbox, feat] = body_feat
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
