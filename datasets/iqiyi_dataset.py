# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 18:58
# @Author  : LegenDong
# @User    : legendong
# @File    : iqiyi_dataset.py
# @Software: PyCharm
import os

from torch.utils import data

from utils import load_face_from_pickle, load_train_gt_from_txt, check_exists, \
    default_face_pre_progress, default_transforms, default_target_transforms, load_head_from_pickle, \
    load_body_from_pickle, load_val_gt_from_txt, retain_noise_in_val

__all__ = ['IQiYiFaceDataset', 'IQiYiHeadDataset', 'IQiYiBodyDataset']

FEAT_PATH = 'feat'

FACE_TRAIN_NAME = 'face_train_v2.pickle'
FACE_VAL_NAME = 'face_val_v2.pickle'
FACE_TEST_NAME_OLD = 'face_test.pickle'
FACE_TEST_NAME = 'face_test_v2.pickle'

HEAD_TRAIN_NAME = 'head_train.pickle'
HEAD_VAL_NAME = 'head_val.pickle'
HEAD_TEST_NAME = 'head_test.pickle'

BODY_TRAIN_NAME = 'body_train.pickle'
BODY_VAL_NAME = 'body_val.pickle'
BODY_TEST_NAME = 'body_test.pickle'

TRAIN_GT_NAME = 'train_gt.txt'
VAL_GT_NAME = 'val_gt.txt'


class IQiYiFaceDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val', 'train+val', 'train+noise', 'test', ]

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_face_pre_progress
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
        elif self.tvt == 'train+val' or self.tvt == 'train+noise':
            self.train_feats_path = os.path.join(self.root, FEAT_PATH, FACE_TRAIN_NAME)
            self.val_feats_path = os.path.join(self.root, FEAT_PATH, FACE_VAL_NAME)
            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FEAT_PATH, FACE_TEST_NAME)
            if not check_exists(self.feats_path):
                self.feats_path = os.path.join(self.root, FEAT_PATH, FACE_TEST_NAME_OLD)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            video_infos = load_face_from_pickle(self.feats_path)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            video_infos = load_face_from_pickle(self.feats_path)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise':
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
                = retain_noise_in_val(self.feats, self.labels, self.video_names, **self.kwargs)
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
        assert tvt in ['train', 'val', 'test']

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_face_pre_progress
        if self.transform is None:
            self.transform = default_transforms
        if self.target_transform is None:
            self.target_transform = default_target_transforms

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, FEAT_PATH, HEAD_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, FEAT_PATH, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, FEAT_PATH, HEAD_VAL_NAME)
            self.gt_path = None
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FEAT_PATH, HEAD_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        video_infos = load_head_from_pickle(self.feats_path)
        if self.tvt == 'train':
            gt_labels = load_train_gt_from_txt(self.gt_path)
        else:
            gt_labels = {}
        self.feats, self.labels, self.video_names \
            = self.pre_progress(video_infos, gt_labels, **self.kwargs)
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
        assert tvt in ['train', 'val', 'test']

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_face_pre_progress
        if self.transform is None:
            self.transform = default_transforms
        if self.target_transform is None:
            self.target_transform = default_target_transforms

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, FEAT_PATH, BODY_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, FEAT_PATH, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, FEAT_PATH, BODY_VAL_NAME)
            self.gt_path = None
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FEAT_PATH, BODY_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        video_infos = load_body_from_pickle(self.feats_path)
        if self.tvt == 'train':
            gt_labels = load_train_gt_from_txt(self.gt_path)
        else:
            gt_labels = {}
        self.feats, self.labels, self.video_names \
            = self.pre_progress(video_infos, gt_labels, **self.kwargs)
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
