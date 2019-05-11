# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 18:58
# @Author  : LegenDong
# @User    : legendong
# @File    : iqiyi_dataset.py
# @Software: PyCharm
import os

from torch.utils import data

import utils as module
from utils import load_face_from_pickle, load_train_gt_from_txt, check_exists, \
    default_pre_progress, default_transforms, default_target_transforms, load_val_gt_from_txt, \
    default_retain_noise_in_val, default_vid_retain_noise_in_val, \
    default_vid_pre_progress, default_vid_transforms, default_vid_target_transforms, default_vid_remove_noise_in_val, \
    default_remove_noise_in_val, load_head_from_pickle, load_body_from_pickle

__all__ = ['IQiYiVidDataset', 'IQiYiFaceDataset', 'IQiYiHeadDataset', 'IQiYiBodyDataset']

FEAT_PATH = 'feat'

FACE_TRAIN_NAME = 'face_train_v2.pickle'
FACE_VAL_NAME = 'face_val_v2.pickle'
FACE_TEST_NAME = 'face_test.pickle'

HEAD_TRAIN_NAME = 'head_train.pickle'
HEAD_VAL_NAME = 'head_val.pickle'
HEAD_TEST_NAME = 'head_test.pickle'

BODY_TRAIN_NAME = 'body_train.pickle'
BODY_VAL_NAME = 'body_val.pickle'
BODY_TEST_NAME = 'body_test.pickle'

TRAIN_GT_NAME = 'train_gt.txt'
VAL_GT_NAME = 'val_gt.txt'

TEMPLATE_LOAD_PICKLE = 'load_{}_from_pickle'


def gen_file_name(mode, tvt):
    if mode == 'face':
        if tvt == 'train':
            file_name = FACE_TRAIN_NAME
        elif tvt == 'val':
            file_name = FACE_VAL_NAME
        else:
            file_name = FACE_TEST_NAME
    elif mode == 'head':
        if tvt == 'train':
            file_name = HEAD_TRAIN_NAME
        elif tvt == 'val':
            file_name = HEAD_VAL_NAME
        else:
            file_name = HEAD_TEST_NAME
    elif mode == 'body':
        if tvt == 'train':
            file_name = BODY_TRAIN_NAME
        elif tvt == 'val':
            file_name = BODY_VAL_NAME
        else:
            file_name = BODY_TEST_NAME
    else:
        raise RuntimeError

    return file_name


class IQiYiVidDataset(data.Dataset):
    def __init__(self, root, tvt='train', modes='face', transform=None, target_transform=None, pre_progress=None,
                 **kwargs):
        modes = modes.split('+')

        assert check_exists(root)
        assert tvt in ['train', 'val', 'train+val', 'train+noise', 'train+val-noise', 'test', ]
        assert 0 < len(modes) < 4
        for sub in modes:
            assert sub in ['face', 'head', 'body']

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.modes = modes
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_vid_pre_progress
        else:
            assert 'vid' in pre_progress.__name__.lower()
        if self.transform is None:
            self.transform = default_vid_transforms
        else:
            assert 'vid' in transform.__name__.lower()
        if self.target_transform is None:
            self.target_transform = default_vid_target_transforms
        else:
            assert 'vid' in target_transform.__name__.lower()

        if self.tvt == 'train':
            self.feats_paths = {}
            for mode in modes:
                self.feats_paths[mode] = os.path.join(self.root, FEAT_PATH, gen_file_name(mode, 'train'))
            self.gt_path = os.path.join(self.root, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_paths = {}
            for mode in modes:
                self.feats_paths[mode] = os.path.join(self.root, FEAT_PATH, gen_file_name(mode, 'val'))
            self.gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            self.train_feats_paths = {}
            self.val_feats_paths = {}
            for mode in modes:
                self.train_feats_paths[mode] = os.path.join(self.root, FEAT_PATH, gen_file_name(mode, 'train'))
                self.val_feats_paths[mode] = os.path.join(self.root, FEAT_PATH, gen_file_name(mode, 'val'))
            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.feats_paths = {}
            for mode in modes:
                self.feats_paths[mode] = os.path.join(self.root, FEAT_PATH, gen_file_name(mode, 'test'))
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            video_infos = {}
            for key, value in self.feats_paths.items():
                assert hasattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                load_func = getattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                video_infos[key] = load_func(value)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            video_infos = {}
            for key, value in self.feats_paths.items():
                assert hasattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                load_func = getattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                video_infos[key] = load_func(value)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            video_infos = {}
            for key, value in self.train_feats_paths.items():
                assert hasattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                load_func = getattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                video_infos[key] = []
                video_infos[key] += load_func(value)
                video_infos[key] += load_func(self.val_feats_paths[key])

            gt_labels = {}
            gt_labels.update(load_train_gt_from_txt(self.train_gt_path))
            gt_labels.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            video_infos = {}
            for key, value in self.feats_paths.items():
                assert hasattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                load_func = getattr(module, TEMPLATE_LOAD_PICKLE.format(key))
                video_infos[key] = load_func(value)
            gt_labels = {}

        self.vid_infos = self.pre_progress(video_infos, gt_labels, **self.kwargs)
        if self.tvt == 'train+noise':
            self.vid_infos = default_vid_retain_noise_in_val(self.vid_infos, **self.kwargs)
        elif self.tvt == 'train+val-noise':
            self.vid_infos = default_vid_remove_noise_in_val(self.vid_infos, **self.kwargs)

        self.length = len(self.vid_infos)

    def __getitem__(self, index):
        vid_info = self.vid_infos[index]

        label = self.target_transform(vid_info, **self.kwargs)
        vid_name = vid_info['video_name']

        if len(self.modes) == 1:
            feat = self.transform(vid_info, self.modes, **self.kwargs)
            return feat, label, vid_name
        elif len(self.modes) == 2:
            feat_1, feat_2 = self.transform(vid_info, self.modes, **self.kwargs)
            return feat_1, feat_2, label, vid_name
        elif len(self.modes) == 3:
            feat_1, feat_2, feat_3 = self.transform(vid_info, self.modes, **self.kwargs)
            return feat_1, feat_2, feat_3, label, vid_name

    def __len__(self):
        return self.length


class IQiYiFaceDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val', 'train+val', 'train+noise', 'train+val-noise', 'test', ]

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_pre_progress
        if self.transform is None:
            self.transform = default_transforms
        if self.target_transform is None:
            self.target_transform = default_target_transforms

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, FEAT_PATH, FACE_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, FEAT_PATH, FACE_VAL_NAME)
            self.gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            self.train_feats_path = os.path.join(self.root, FEAT_PATH, FACE_TRAIN_NAME)
            self.val_feats_path = os.path.join(self.root, FEAT_PATH, FACE_VAL_NAME)
            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FEAT_PATH, FACE_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            video_infos = load_face_from_pickle(self.feats_path)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            video_infos = load_face_from_pickle(self.feats_path)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            video_infos = []
            video_infos += load_face_from_pickle(self.train_feats_path)
            video_infos += load_face_from_pickle(self.val_feats_path)

            gt_labels = {}
            gt_labels.update(load_train_gt_from_txt(self.train_gt_path))
            gt_labels.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            video_infos = load_face_from_pickle(self.feats_path)
            gt_labels = {}

        self.feats, self.labels, self.video_names \
            = self.pre_progress(video_infos, gt_labels, **self.kwargs)
        if self.tvt == 'train+noise':
            self.feats, self.labels, self.video_names \
                = default_retain_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
        elif self.tvt == 'train+val-noise':
            self.feats, self.labels, self.video_names \
                = default_remove_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
        self.length = len(self.feats)

        assert len(self.feats) == len(self.labels)
        assert len(self.feats) == len(self.video_names)

    def __getitem__(self, index):
        feat = self.feats[index]
        label = self.labels[index]
        video_name = self.video_names[index]

        feat = self.transform(feat, **self.kwargs)
        label = self.target_transform(label, **self.kwargs)

        return feat, label, video_name

    def __len__(self):
        return self.length


class IQiYiHeadDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val', 'train+val', 'train+noise', 'train+val-noise', 'test', ]

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_pre_progress
        else:
            assert not ('face' in pre_progress.__name__.lower())
        if self.transform is None:
            self.transform = default_transforms
        if self.target_transform is None:
            self.target_transform = default_target_transforms

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, FEAT_PATH, HEAD_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, FEAT_PATH, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, FEAT_PATH, HEAD_VAL_NAME)
            self.gt_path = os.path.join(self.root, FEAT_PATH, VAL_GT_NAME)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            self.train_feats_path = os.path.join(self.root, FEAT_PATH, HEAD_TRAIN_NAME)
            self.val_feats_path = os.path.join(self.root, FEAT_PATH, HEAD_VAL_NAME)
            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FEAT_PATH, HEAD_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            video_infos = load_head_from_pickle(self.feats_path)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            video_infos = load_head_from_pickle(self.feats_path)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            video_infos = []
            video_infos += load_head_from_pickle(self.train_feats_path)
            video_infos += load_head_from_pickle(self.val_feats_path)

            gt_labels = {}
            gt_labels.update(load_train_gt_from_txt(self.train_gt_path))
            gt_labels.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            video_infos = load_head_from_pickle(self.feats_path)
            gt_labels = {}

        self.feats, self.labels, self.video_names \
            = self.pre_progress(video_infos, gt_labels, is_face=False, **self.kwargs)
        if self.tvt == 'train+noise':
            self.feats, self.labels, self.video_names \
                = default_retain_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
        elif self.tvt == 'train+val-noise':
            self.feats, self.labels, self.video_names \
                = default_remove_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
        self.length = len(self.feats)

        assert len(self.feats) == len(self.labels)
        assert len(self.feats) == len(self.video_names)

    def __getitem__(self, index):
        feat = self.feats[index]
        label = self.labels[index]
        video_name = self.video_names[index]

        feat = self.transform(feat, **self.kwargs)
        label = self.target_transform(label, **self.kwargs)

        return feat, label, video_name

    def __len__(self):
        return self.length


class IQiYiBodyDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val', 'train+val', 'train+noise', 'train+val-noise', 'test', ]

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_pre_progress
        else:
            assert not ('face' in pre_progress.__name__.lower())
        if self.transform is None:
            self.transform = default_transforms
        if self.target_transform is None:
            self.target_transform = default_target_transforms

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, FEAT_PATH, BODY_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, FEAT_PATH, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, FEAT_PATH, BODY_VAL_NAME)
            self.gt_path = os.path.join(self.root, FEAT_PATH, VAL_GT_NAME)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            self.train_feats_path = os.path.join(self.root, FEAT_PATH, BODY_TRAIN_NAME)
            self.val_feats_path = os.path.join(self.root, FEAT_PATH, BODY_VAL_NAME)
            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FEAT_PATH, BODY_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            video_infos = load_body_from_pickle(self.feats_path)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            video_infos = load_body_from_pickle(self.feats_path)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            video_infos = []
            video_infos += load_body_from_pickle(self.train_feats_path)
            video_infos += load_body_from_pickle(self.val_feats_path)

            gt_labels = {}
            gt_labels.update(load_train_gt_from_txt(self.train_gt_path))
            gt_labels.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            video_infos = load_body_from_pickle(self.feats_path)
            gt_labels = {}

        self.feats, self.labels, self.video_names \
            = self.pre_progress(video_infos, gt_labels, is_face=False, **self.kwargs)
        if self.tvt == 'train+noise':
            self.feats, self.labels, self.video_names \
                = default_retain_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
        elif self.tvt == 'train+val-noise':
            self.feats, self.labels, self.video_names \
                = default_remove_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
        self.length = len(self.feats)

        assert len(self.feats) == len(self.labels)
        assert len(self.feats) == len(self.video_names)

    def __getitem__(self, index):
        feat = self.feats[index]
        label = self.labels[index]
        video_name = self.video_names[index]

        feat = self.transform(feat, **self.kwargs)
        label = self.target_transform(label, **self.kwargs)

        return feat, label, video_name

    def __len__(self):
        return self.length
