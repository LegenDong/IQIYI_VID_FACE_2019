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
import uuid
import cv2

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
           'default_image_pre_progress', 'default_image_transforms', 'default_image_target_transforms', 'crop_image',
           'prepare_device', 'default_image_remove_noise_in_val', 'default_scene_pre_progress', 'aug_vid_pre_progress',
           'default_scene_transforms', 'default_scene_target_transforms', 'default_scene_remove_noise_in_val',
           'sep_cat_qds_select_vid_transforms', 'merge_multi_view_result', 'get_mask_index', 'load_scene_infos',
           'default_scene_feat_pre_progress', 'default_scene_feat_remove_noise', 'default_sep_scene_feat_transforms',
           'default_scene_feat_target_transforms', 'split_name_by_l2norm', 'default_fine_tune_pre_progress',
           'default_fine_tune_transforms', 'default_fine_tune_target_transforms', 'adjust_learning_rate',
           'default_sep_select_scene_feat_transforms', 'info_vid_pre_progress', 'clamp_vid_pre_progress',
           'sep_cat_qds_mixup_vid_transforms', 'sep_cat_info_vid_transforms']

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
    frame_ids = []
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
            frame_ids.append(frame_id)
    return file_paths, labels, video_names, bboxes, frame_ids


def default_scene_pre_progress(tvt, image_root, num_frame=10, candidate_names=None, **kwargs):
    image_paths = []
    video_names = []
    image_indexes = []

    all_video_names = os.listdir(image_root)
    for video_name in all_video_names:
        if tvt in video_name.lower() and (candidate_names is None or video_name in candidate_names):
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


def info_vid_pre_progress(video_infos, gt_infos, meta_info, root='/data/materials/video/', **kwargs):
    print('info_vid_pre_process: not complete, better not use it?!')
    exit()
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

    meta_root = '/data/materials/feat/meta_info_{}.pickle'.format(meta_info)
    print(meta_root)

    if check_exists(meta_root):
        print('meta file exits')
        with open(meta_root, 'rb') as f:
            meta_infos = pickle.load(f)
        f.close()

        for key, value in vid_infos.items():
            meta_info = meta_infos[key]
            vid_infos[key]['fps'] = meta_info['fps']
            vid_infos[key]['width'] = meta_info['width']
            vid_infos[key]['height'] = meta_info['height']
    else:
        print('meta file does not exit')
        meta_infos = {}

        for key, value in vid_infos.items():
            vid_path = os.path.join(root, key + '.mp4')
            cap = cv2.VideoCapture(vid_path)
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            vid_infos[key]['fps'] = fps
            vid_infos[key]['width'] = width
            vid_infos[key]['height'] = height

            meta_info = {}
            meta_info['fps'] = fps
            meta_info['width'] = width
            meta_info['height'] = height
            meta_infos[key] = meta_info

            cap.release()

        print('saving meta file')
        with open(meta_root, 'wb') as f:
            pickle.dump(meta_infos, f)
        f.close()

    return list(vid_infos.values())


def clamp_vid_pre_progress(video_infos, gt_infos, face_l2_threshold=15, face_d_shreshold=0.9, **kwargs):
    vid_infos = default_vid_pre_progress(video_infos, gt_infos)
    print(len(vid_infos))
    to_dels = []
    for index, vid_info in enumerate(vid_infos):
        if 'face' in vid_info.keys():
            frames = vid_info['face']
            vid_info['face'] = [frame for frame in frames if np.linalg.norm(frame['feat']) >= face_l2_threshold \
                                and frame['det_score'] >= face_d_shreshold]
            if len(vid_info['face']) == 0:
                to_dels.append(index)
    for to_del in sorted(to_dels, reverse=True):
        del vid_infos[to_del]
    print(len(vid_infos))
    return vid_infos


def aug_vid_pre_progress(video_infos, gt_infos, aug_num_vid=10, aug_num_frame=50, only_train=True, **kwargs):
    assert aug_num_vid >= 0
    vid_infos = {}
    no_frame = []
    for key, values in video_infos.items():
        for value in values:
            frame_infos = value['frame_infos']
            if len(frame_infos) > 0:
                vid_infos.setdefault(value['video_name'], {})[key] = frame_infos
            else:
                no_frame.append(value['video_name'])
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

    if aug_num_vid > 0:
        id_infos = {}
        for key, value in gt_infos.items():
            if not only_train or 'TRAIN' in key:
                id_infos.setdefault(value, []).append(key)

        count_aug = 1
        for key, values in id_infos.items():
            values = list(set(values).difference(set(to_dels)))
            values = list(set(values).difference(set(no_frame)))
            num_vid = len(values)
            if num_vid == 0:
                continue
            vids = [vid_infos[value] for value in values]
            for i in range(aug_num_vid):
                new_vid = dict()
                new_vid['label'] = key
                new_vid['video_name'] = 'IQIYI_VID_AUG_' + str.zfill(str(count_aug), 7)
                index = [np.random.randint(0, num_vid) for _ in range(aug_num_frame)]
                choose_vids = [vids[i] for i in index]

                for moda in video_infos.keys():
                    feat_moda = []
                    for vid in choose_vids:
                        choose_frame = np.random.choice(vid[moda], 1)
                        feat_moda += list(choose_frame)
                    assert len(feat_moda) == aug_num_frame
                    new_vid[moda] = feat_moda

                vid_infos[new_vid['video_name']] = new_vid
                count_aug += 1
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
        if ('TRAIN' in vid_info['video_name'] or 'AUG' in vid_info['video_name']) or \
                (vid_info['label'] == 0 and 'VAL' in vid_info['video_name'] and random.uniform(0, 1) < pr):
            idx_list.append(idx)
    return [vid_infos[idx] for idx in idx_list]


def default_vid_remove_noise_in_val(vid_infos, **kwargs):
    idx_list = []
    for idx, vid_info in enumerate(vid_infos):
        if ('TRAIN' in vid_info['video_name']) \
                or (vid_info['label'] != 0 and 'VAL' in vid_info['video_name']) \
                or (vid_info['label'] != 0 and 'AUG' in vid_info['video_name']):
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


def default_image_remove_noise_in_val(file_paths, labels, video_names, bboxes, frame_ids, **kwargs):
    assert len(file_paths) == len(labels)
    assert len(labels) == len(video_names)
    assert len(video_names) == len(bboxes)
    assert len(bboxes) == len(frame_ids)

    idx_list = []
    for label_idx, label in enumerate(labels):
        if 'TRAIN' in video_names[label_idx] or (label != 0 and 'VAL' in video_names[label_idx]):
            idx_list.append(label_idx)
    file_paths = [file_paths[idx] for idx in idx_list]
    labels = [labels[idx] for idx in idx_list]
    video_names = [video_names[idx] for idx in idx_list]
    bboxes = [bboxes[idx] for idx in idx_list]
    frame_ids = [frame_ids[idx] for idx in idx_list]

    return file_paths, labels, video_names, bboxes, frame_ids


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


def sep_cat_qds_select_vid_transforms(vid_info, modes, num_frame=15, mask_index=None, **kwargs):
    result = []
    for mode in modes:
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frames_info in frames_infos:
            temp_feat = frames_info['feat']
            if mask_index is not None:
                temp_feat = temp_feat[mask_index]
            if mode == 'face':
                temp_feat = np.append(temp_feat, frames_info['quality_score'] / 100.)
            else:
                temp_feat = np.append(temp_feat, frames_info['det_score'])
            temp_feat = np.append(temp_feat, frames_info['det_score'])
            temp_feats.append(temp_feat)
        feats = np.array(temp_feats)
        result.append(torch.from_numpy(feats).float())
    return result


def sep_cat_info_vid_transforms(vid_info, modes, num_frame=15,
                                q_norm_value=100., use_meta=False, w_norm=1000., h_norm=1000., **kwargs):
    result = []
    for mode in modes:
        vid_fps = vid_info['fps']
        vid_width = vid_info['width']
        vid_height = vid_info['height']
        frames_infos = vid_info[mode]
        if len(frames_infos) < num_frame:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=True)
        else:
            frames_infos = np.random.choice(frames_infos, num_frame, replace=False)
        temp_feats = []
        for frame_info in frames_infos:
            feat = frame_info['feat']
            if use_meta:
                feat = np.append(feat, vid_fps)
                feat = np.append(feat, vid_width / w_norm)
                feat = np.append(feat, vid_height / h_norm)

            if mode == 'face':
                feat = np.append(feat, frame_info['quality_score'] / q_norm_value)
                [x1, y1, x2, y2] = frame_info['bbox']
                bbox_rate = ((x1 - x2) * (y1 - y2)) / (vid_width * vid_height)
                feat = np.append(feat, bbox_rate)
            else:
                feat = np.append(feat, frame_info['det_score'])

            feat = np.append(feat, frame_info['det_score'])
            temp_feats.append(feat)
        feats = np.array(temp_feats)
        result.append(torch.from_numpy(feats).float())
    return result


def sep_cat_qds_mixup_vid_transforms(vid_info, modes, num_frame=15, norm_value=100., mixup_rate=0.5, **kwargs):
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
        feats1 = np.array(temp_feats)

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
        feats2 = np.array(temp_feats)

        l = np.random.beta(mixup_rate, mixup_rate)
        feats = l * feats1 + (1 - l) * feats2

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


def crop_image(image_data, bbox, ratio):
    bbox = _padding_bbox(bbox, ratio)
    image_data = image_data.crop(bbox)
    return image_data


def _padding_bbox(bbox, ratio=1.):
    x1, y1, x2, y2 = bbox
    x_padding = (ratio - 1) * (x2 - x1) / 2
    y_padding = (ratio - 1) * (y2 - y1) / 2
    bbox = (x1 - x_padding, y1 - y_padding, x2 + x_padding, y2 + y_padding)

    return bbox


def default_image_transforms(image_data, mean_bgr, augm_func, **kwargs):
    image_data = augm_func(image_data)
    image_data = np.array(image_data, dtype=np.uint8)
    image_data = _trans_img(image_data, mean_bgr)

    return image_data


def default_scene_transforms(image_data, augm_func, **kwargs):
    image_data = augm_func(image_data)
    return image_data


def _trans_img(img, mean_bgr):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img


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
            assert (feat.dtype == np.float16 and (feat.shape[0] == 512 or feat.shape[0] == 2048))

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


def default_sep_scene_feat_transforms(frame_infos, **kwargs):
    temp_feats = []
    for frame_info in frame_infos:
        temp_feats.append(frame_info[1])

    feats = torch.from_numpy(np.array(temp_feats)).float()

    return feats


def default_sep_select_scene_feat_transforms(frame_infos, mask_index=None, **kwargs):
    temp_feats = []

    for frame_info in frame_infos:
        temp_feat = frame_info[1][mask_index]
        temp_feats.append(temp_feat)

    feats = torch.from_numpy(np.array(temp_feats)).float()

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


def adjust_learning_rate(optimizer, epoch_idx, warm_up_epochs, learning_rate):
    lr = learning_rate * (epoch_idx + 1) / (warm_up_epochs + 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
