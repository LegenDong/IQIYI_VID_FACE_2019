# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 18:58
# @Author  : LegenDong
# @User    : legendong
# @File    : iqiyi_dataset.py
# @Software: PyCharm
import os

from torch.utils import data

from utils import load_face_from_pickle, load_train_gt_from_txt, check_exists, \
    default_pre_progress, default_transforms, default_target_transforms

__all__ = ['IQiYiFaceDataset']

FACE_TRAIN_NAME = 'face_train.pickle'
FACE_VAL_NAME = 'face_val.pickle'
FACE_TEST_NAME = 'face_test.pickle'

TRAIN_GT_NAME = 'train_gt.txt'
VAL_GT_NAME = 'val_gt.txt'


class IQiYiFaceDataset(data.Dataset):
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
            self.pre_progress = default_pre_progress
        if self.transform is None:
            self.transform = default_transforms
        if self.target_transform is None:
            self.target_transform = default_target_transforms

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, FACE_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, FACE_VAL_NAME)
            self.gt_path = None
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, FACE_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        video_infos = load_face_from_pickle(self.feats_path)
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
