# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:28
# @Author  : LegenDong
# @User    : legendong
# @File    : utils.py
# @Software: PyCharm
import logging
import os
import pickle
import uuid

import numpy as np
import torch

__all__ = ['init_logging', 'check_exists', 'load_train_gt_from_txt', 'load_val_gt_from_txt', 'load_face_from_pickle',
           'default_get_result', 'default_face_scene_target_transforms', 'save_model', 'topk_func',
           'default_vid_transforms',
           'prepare_device', 'default_scene_pre_progress', 'default_scene_transforms',
           'default_scene_target_transforms', 'default_scene_remove_noise_in_val', 'merge_multi_view_result',
           'get_mask_index', 'load_scene_infos', 'default_scene_feat_pre_progress', 'default_scene_feat_remove_noise',
           'default_scene_feat_transforms', 'default_scene_feat_target_transforms', 'split_name_by_l2norm',
           'default_fine_tune_pre_progress', 'default_fine_tune_transforms', 'default_fine_tune_target_transforms',
           'default_sep_select_scene_feat_transforms', 'default_face_scene_remove_noise_in_val',
           'default_face_scene_pre_progress', 'sep_cat_qds_face_scene_transforms',
           'sep_cat_qds_select_face_scene_transforms']

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
            logger.warning('check_exists: {} not exist'.format(file_path))
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


def default_scene_pre_progress(tvt, image_root, num_frame=1, **kwargs):
    image_paths = []
    video_names = []
    image_indexes = []

    all_video_names = os.listdir(image_root)
    for video_name in all_video_names:
        if tvt in video_name.lower():
            video_image_root = os.path.join(image_root, video_name)
            all_image_names = os.listdir(video_image_root)
            for float_idx in np.linspace(1, len(all_image_names), num_frame, endpoint=True):
                int_index = np.math.floor(float_idx) - 1

                image_name = all_image_names[int_index]
                image_path = os.path.join(video_image_root, image_name)
                image_index = int(os.path.splitext(image_name)[0])

                image_paths.append(image_path)
                video_names.append(video_name)
                image_indexes.append(image_index)

    return image_paths, video_names, image_indexes


def default_scene_remove_noise_in_val(file_paths, labels, video_names, **kwargs):
    assert len(file_paths) == len(labels)
    assert len(labels) == len(video_names)

    idx_list = []
    for label_idx, label in enumerate(labels):
        if 'TRAIN' in video_names[label_idx] or (label != 0 and 'VAL' in video_names[label_idx]):
            idx_list.append(label_idx)
    file_paths = [file_paths[idx] for idx in idx_list]
    labels = [labels[idx] for idx in idx_list]
    video_names = [video_names[idx] for idx in idx_list]

    return file_paths, labels, video_names


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


def default_scene_transforms(image_data, augm_func, **kwargs):
    image_data = augm_func(image_data)
    return image_data


def default_face_scene_target_transforms(label, **kwargs):
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def default_scene_target_transforms(label, **kwargs):
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


def merge_multi_view_result(result_root, is_save=True):
    assert check_exists(result_root)
    pickle_name_list = os.listdir(result_root)

    output_num = 0
    last_name_list = None
    output_sum = .0
    for pickle_name in pickle_name_list:
        logger.info('load pickle file from {}'.format(pickle_name))
        load_path = os.path.join(result_root, pickle_name)
        if check_exists(load_path):
            with open(os.path.join(result_root, pickle_name), 'rb') as fin:
                pickle_file_data = pickle.load(fin, encoding='bytes')
                output_num += pickle_file_data[0]
                output_sum += pickle_file_data[2]
                if last_name_list is not None and last_name_list != pickle_file_data[1]:
                    logger.warning('the name list in {} is different from before'.format(pickle_name))
                else:
                    last_name_list = pickle_file_data[1]
            os.remove(load_path)
            logger.info('pickle file {} has been removed'.format(pickle_name))

    pickle_file_data = (output_num, last_name_list, output_sum)

    if is_save:
        save_name = 'multi_view_face_{}.pickle'.format(uuid.uuid1())
        logger.info('save pickle {} in {} with output num is {}'.format(save_name, result_root, output_num))
        with open(os.path.join(result_root, save_name), 'wb') as fout:
            pickle.dump(pickle_file_data, fout)

    return pickle_file_data


def get_mask_index(seed, feat_length, split_num):
    feat_idxes = list(range(feat_length))
    split_length = feat_length // split_num

    assert split_length * split_num == feat_length

    all_splits = [feat_idxes[i:i + split_length] for i in range(0, len(feat_idxes), split_length)]
    mask_index = []
    for idx in range(len(all_splits)):
        if idx != seed:
            mask_index += all_splits[idx]
    return mask_index


def load_scene_infos(file_path):
    assert check_exists(file_path)
    with open(file_path, 'rb') as fin:
        scene_infos = pickle.load(fin, encoding='bytes')
    return scene_infos


def default_scene_feat_pre_progress(scene_infos, gt_infos, **kwargs):
    all_frame_infos = []
    all_labels = []
    all_video_names = []

    for video_name, frame_infos in scene_infos.items():
        all_labels.append(gt_infos.get(video_name, 0))
        all_frame_infos.append(frame_infos)
        all_video_names.append(video_name)
    return all_frame_infos, all_labels, all_video_names


def default_scene_feat_remove_noise(frame_infos, labels, video_names, **kwargs):
    assert len(frame_infos) == len(labels)
    assert len(labels) == len(video_names)
    idx_list = []
    for label_idx, label in enumerate(labels):
        if 'TRAIN' in video_names[label_idx] or (label != 0 and 'VAL' in video_names[label_idx]):
            idx_list.append(label_idx)

    frame_infos = [frame_infos[idx] for idx in idx_list]
    labels = [labels[idx] for idx in idx_list]
    video_names = [video_names[idx] for idx in idx_list]

    return frame_infos, labels, video_names


def default_scene_feat_transforms(frame_infos, **kwargs):
    temp_feats = []
    for frame_info in frame_infos:
        temp_feats.append(frame_info[1])

    feats = torch.from_numpy(np.array(temp_feats).reshape(-1)).float()

    return feats


def default_sep_select_scene_feat_transforms(frame_infos, mask_index=None, **kwargs):
    temp_feats = []

    for frame_info in frame_infos:
        temp_feat = frame_info[1][mask_index]
        temp_feats.append(temp_feat)

    feats = torch.from_numpy(np.array(temp_feats).reshape(-1)).float()

    return feats


def default_scene_feat_target_transforms(label, **kwargs):
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def split_name_by_l2norm(file_path, split_points):
    if not isinstance(split_points, list):
        if isinstance(split_points, tuple):
            split_points = list(split_points)
        else:
            split_points = [split_points]
    split_points.sort()
    split_names = [[] for _ in range(len(split_points) + 1)]
    video_infos = load_face_from_pickle(file_path)

    for video_info in video_infos:
        feat_list = []
        frame_infos = video_info['frame_infos']
        if len(frame_infos) == 0:
            split_names[0].append(video_info['video_name'])
            continue
        for frame_info in frame_infos:
            feat_list.append(frame_info['feat'])
        feats_np = np.array(feat_list)
        norm_value = np.mean(np.linalg.norm(feats_np, axis=1))
        for split_idx, split_point in enumerate(split_points):
            if norm_value < split_point:
                split_names[split_idx + 1].append(video_info['video_name'])
                break
    logger.info('split data set by {} over.'.format(' '.join([str(point) for point in split_points])))

    return split_names


def default_fine_tune_pre_progress(gt_infos, image_root, **kwargs):
    image_paths = []
    labels = []
    video_names = []

    for video_name, label in gt_infos.items():
        video_root = os.path.join(image_root, video_name, )
        image_list = os.listdir(video_root)

        temp_list = [os.path.join(video_root, image_list[idx])
                     for idx in [0, len(image_list) // 2, len(image_list) - 1]]
        image_paths.append(temp_list)
        labels.append(label)
        video_names.append(video_name)

    return image_paths, labels, video_names


def default_fine_tune_transforms(image_data, augm_func, **kwargs):
    image_data = augm_func(image_data)
    return image_data


def default_fine_tune_target_transforms(label, **kwargs):
    label_np = np.array(label)
    label_torch = torch.from_numpy(label_np).long()
    return label_torch


def default_face_scene_pre_progress(face_feat_infos, scene_feat_infos, gt_infos, **kwargs):
    vid_infos = {}
    for face_feat_info in face_feat_infos:
        frame_infos = face_feat_info['frame_infos']
        video_name = face_feat_info['video_name']
        if len(frame_infos) > 0:
            vid_infos.setdefault(video_name, {})['face'] = frame_infos
            vid_infos.setdefault(video_name, {})['scene'] = scene_feat_infos[video_name]
            vid_infos.setdefault(video_name, {})['label'] = gt_infos.get(video_name, 0)
            vid_infos.setdefault(video_name, {})['video_name'] = video_name
    return list(vid_infos.values())


def sep_cat_qds_face_scene_transforms(vid_info, num_frame=15, norm_value=100., **kwargs):
    result = []
    face_frame_infos = vid_info['face']
    if len(face_frame_infos) < num_frame:
        frames_infos = np.random.choice(face_frame_infos, num_frame, replace=True)
    else:
        frames_infos = np.random.choice(face_frame_infos, num_frame, replace=False)
    face_feats = []
    for frame_info in frames_infos:
        feat = frame_info['feat']
        feat = np.append(feat, frame_info['quality_score'] / norm_value)
        feat = np.append(feat, frame_info['det_score'])
        face_feats.append(feat)
    feats = np.array(face_feats)
    result.append(torch.from_numpy(feats).float())

    scene_frame_infos = vid_info['scene']
    scene_feats = []
    for frame_info in scene_frame_infos:
        temp_feat = frame_info[1]
        scene_feats.append(temp_feat)
    feats = np.array(scene_feats).reshape(-1)
    result.append(torch.from_numpy(feats).float())

    return result


def sep_cat_qds_select_face_scene_transforms(vid_info, face_mask=None, scene_mask=None, num_frame=15, norm_value=100.,
                                             **kwargs):
    result = []
    face_frame_infos = vid_info['face']
    if len(face_frame_infos) < num_frame:
        frames_infos = np.random.choice(face_frame_infos, num_frame, replace=True)
    else:
        frames_infos = np.random.choice(face_frame_infos, num_frame, replace=False)
    face_feats = []
    for frame_info in frames_infos:
        feat = frame_info['feat']
        if face_mask is not None:
            feat = feat[face_mask]
        feat = np.append(feat, frame_info['quality_score'] / norm_value)
        feat = np.append(feat, frame_info['det_score'])
        face_feats.append(feat)
    feats = np.array(face_feats)
    result.append(torch.from_numpy(feats).float())

    scene_frame_infos = vid_info['scene']
    scene_feats = []
    for frame_info in scene_frame_infos:
        feat = frame_info[1]
        if scene_mask is not None:
            feat = feat[scene_mask]
        scene_feats.append(feat)
    feats = np.array(scene_feats).reshape(-1)
    result.append(torch.from_numpy(feats).float())

    return result


def default_face_scene_remove_noise_in_val(vid_infos, **kwargs):
    idx_list = []
    for idx, vid_info in enumerate(vid_infos):
        if ('TRAIN' in vid_info['video_name']) \
                or (vid_info['label'] != 0 and 'VAL' in vid_info['video_name']) \
                or (vid_info['label'] != 0 and 'AUG' in vid_info['video_name']):
            idx_list.append(idx)
    return [vid_infos[idx] for idx in idx_list]
