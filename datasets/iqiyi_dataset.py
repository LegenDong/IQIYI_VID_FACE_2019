# -*- coding: utf-8 -*-
# @Time    : 2019/4/21 18:58
# @Author  : LegenDong
# @User    : legendong
# @File    : iqiyi_dataset.py
# @Software: PyCharm
import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from utils import load_face_from_pickle, load_train_gt_from_txt, check_exists, default_face_scene_target_transforms, \
    load_val_gt_from_txt, \
    default_scene_pre_progress, default_scene_transforms, \
    default_scene_target_transforms, load_scene_infos, default_scene_feat_pre_progress, \
    default_scene_feat_remove_noise, default_scene_feat_transforms, default_scene_feat_target_transforms, \
    default_fine_tune_pre_progress, default_fine_tune_transforms, default_fine_tune_target_transforms, \
    default_face_scene_pre_progress, sep_cat_qds_face_scene_transforms, default_face_scene_remove_noise_in_val

__all__ = ['IQiYiExtractSceneDataset', 'IQiYiSceneFeatDataset', 'IQiYiFineTuneSceneDataset', 'IQiYiFaceSceneDataset', ]

FEAT_PATH = 'feat'
IMAGE_PATH = 'img'

FACE_TRAIN_NAME = 'face_train_v2.pickle'
FACE_VAL_NAME = 'face_val_v2.pickle'
FACE_TEST_NAME = 'face_test.pickle'

SCENE_TRAIN_NAME = 'scene_infos_train.pickle'
SCENE_VAL_NAME = 'scene_infos_val.pickle'
SCENE_TEST_NAME = 'scene_infos_test.pickle'

TRAIN_GT_NAME = 'train_gt.txt'
VAL_GT_NAME = 'val_gt.txt'


class IQiYiExtractSceneDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, image_root=None,
                 **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val', 'test', ]

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs
        self.image_root = os.path.join(self.root, IMAGE_PATH) \
            if (image_root is None or not check_exists(image_root)) else image_root

        # get the code from https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py
        self.augm_func = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.pre_progress is None:
            self.pre_progress = default_scene_pre_progress
        else:
            assert 'scene' in pre_progress.__name__.lower()
        if self.transform is None:
            self.transform = default_scene_transforms
        else:
            assert 'scene' in transform.__name__.lower()
        if self.target_transform is None:
            self.target_transform = default_scene_target_transforms
        else:
            assert 'scene' in target_transform.__name__.lower()

        self._init_feat_labels()

    def _init_feat_labels(self):
        self.image_paths, self.video_names, self.image_indexes \
            = self.pre_progress(self.tvt, self.image_root, **self.kwargs)
        self.length = len(self.image_paths)
        print(self.length)

        assert len(self.image_paths) == len(self.video_names)
        assert len(self.video_names) == len(self.image_indexes)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        video_name = self.video_names[index]
        image_index = self.image_indexes[index]
        image_data = Image.open(image_path).convert('RGB')
        image_data = self.transform(image_data, self.augm_func)
        return image_data, video_name, image_index

    def __len__(self):
        return self.length


class IQiYiSceneFeatDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val', 'train+val', 'train+val-noise', 'test', ]

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_scene_feat_pre_progress
        else:
            assert 'scene_feat' in pre_progress.__name__.lower()
        if self.transform is None:
            self.transform = default_scene_feat_transforms
        else:
            assert 'scene_feat' in transform.__name__.lower()
        if self.target_transform is None:
            self.target_transform = default_scene_feat_target_transforms
        else:
            assert 'scene_feat' in target_transform.__name__.lower()

        if self.tvt == 'train':
            self.feats_path = os.path.join(self.root, SCENE_TRAIN_NAME)
            self.gt_path = os.path.join(self.root, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.feats_path = os.path.join(self.root, SCENE_VAL_NAME)
            self.gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'train+val' or self.tvt == 'train+val-noise':
            self.train_feats_path = os.path.join(self.root, SCENE_TRAIN_NAME)
            self.val_feats_path = os.path.join(self.root, SCENE_VAL_NAME)

            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.feats_path = os.path.join(self.root, SCENE_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            scene_infos = load_scene_infos(self.feats_path)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            scene_infos = load_scene_infos(self.feats_path)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+noise' or self.tvt == 'train+val-noise':
            scene_infos = {}
            scene_infos.update(load_scene_infos(self.train_feats_path))
            scene_infos.update(load_scene_infos(self.val_feats_path))

            gt_labels = {}
            gt_labels.update(load_train_gt_from_txt(self.train_gt_path))
            gt_labels.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            scene_infos = load_scene_infos(self.feats_path)
            gt_labels = {}

        self.frame_infos, self.labels, self.video_names \
            = self.pre_progress(scene_infos, gt_labels, **self.kwargs)
        if self.tvt == 'train+val-noise':
            self.frame_infos, self.labels, self.video_names \
                = default_scene_feat_remove_noise(self.frame_infos, self.labels, self.video_names, **self.kwargs)
        self.length = len(self.frame_infos)

        assert len(self.frame_infos) == len(self.labels)
        assert len(self.frame_infos) == len(self.video_names)

    def __getitem__(self, index):
        frame_info = self.frame_infos[index]
        label = self.labels[index]
        video_name = self.video_names[index]

        feat = self.transform(frame_info, **self.kwargs)
        label = self.target_transform(label, **self.kwargs)

        return feat, label, video_name

    def __len__(self):
        return self.length


class IQiYiFineTuneSceneDataset(data.Dataset):
    def __init__(self, root, tvt='train', transform=None, target_transform=None, pre_progress=None, image_root=None,
                 **kwargs):
        assert check_exists(root)
        assert tvt in ['train', 'val-noise', 'train+val-noise']

        self.root = os.path.expanduser(root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs
        self.image_root = os.path.join(self.root, IMAGE_PATH) \
            if (image_root is None or not check_exists(image_root)) else image_root

        self.augm_func_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.augm_func_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.is_val = False

        if self.pre_progress is None:
            self.pre_progress = default_fine_tune_pre_progress
        else:
            assert 'fine_tune' in pre_progress.__name__.lower()
        if self.transform is None:
            self.transform = default_fine_tune_transforms
        else:
            assert 'fine_tune' in transform.__name__.lower()
        if self.target_transform is None:
            self.target_transform = default_fine_tune_target_transforms
        else:
            assert 'fine_tune' in target_transform.__name__.lower()

        if self.tvt == 'train':
            self.gt_path = os.path.join(self.root, FEAT_PATH, TRAIN_GT_NAME)
        elif self.tvt == 'val-noise':
            self.gt_path = os.path.join(self.root, FEAT_PATH, VAL_GT_NAME)
        elif self.tvt == 'train+val-noise':
            self.train_gt_path = os.path.join(self.root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.root, VAL_GT_NAME)

        self._init_feat_labels()

    def _init_feat_labels(self):

        if self.tvt == 'train':
            gt_infos = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val-noise':
            gt_infos = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val-noise':
            gt_infos = {}
            gt_infos.update(load_train_gt_from_txt(self.train_gt_path))
            gt_infos.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            raise RuntimeError

        self.image_paths, self.labels, self.video_names \
            = self.pre_progress(gt_infos, self.image_root, **self.kwargs)
        self.length = len(self.image_paths)

        assert len(self.image_paths) == len(self.labels)
        assert len(self.labels) == len(self.video_names)

    def set_val(self, is_val):
        assert isinstance(is_val, bool)
        self.is_val = is_val

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        label = self.labels[index]
        video_name = self.video_names[index]
        image_data_list = []
        for image_path in image_paths:
            image_data = Image.open(image_path).convert('RGB')
            image_data = self.transform(image_data, self.augm_func_val if self.is_val else self.augm_func_train)
            image_data_list.append(image_data.view(1, *image_data.size()))
        images_data = torch.cat(image_data_list, dim=0)
        label = self.target_transform(label)

        return images_data, label, video_name

    def __len__(self):
        return self.length


class IQiYiFaceSceneDataset(data.Dataset):
    def __init__(self, face_root, scene_root, tvt='train', transform=None, target_transform=None, pre_progress=None,
                 **kwargs):
        assert check_exists(face_root)
        assert check_exists(scene_root)
        assert tvt in ['train', 'val', 'train+val', 'train+val-noise', 'test', ]

        self.face_root = os.path.expanduser(face_root)
        self.scene_root = os.path.expanduser(scene_root)
        self.tvt = tvt
        self.transform = transform
        self.target_transform = target_transform
        self.pre_progress = pre_progress
        self.kwargs = kwargs

        if self.pre_progress is None:
            self.pre_progress = default_face_scene_pre_progress
        if self.transform is None:
            self.transform = sep_cat_qds_face_scene_transforms
        if self.target_transform is None:
            self.target_transform = default_face_scene_target_transforms

        if self.tvt == 'train':
            self.face_feats_path = os.path.join(self.face_root, FEAT_PATH, FACE_TRAIN_NAME)
            self.scene_feats_path = os.path.join(self.scene_root, SCENE_TRAIN_NAME)
            self.gt_path = os.path.join(self.face_root, TRAIN_GT_NAME)
        elif self.tvt == 'val':
            self.face_feats_path = os.path.join(self.face_root, FEAT_PATH, FACE_VAL_NAME)
            self.scene_feats_path = os.path.join(self.scene_root, SCENE_VAL_NAME)
            self.gt_path = os.path.join(self.face_root, VAL_GT_NAME)
        elif self.tvt == 'train+val' or self.tvt == 'train+val-noise':
            self.train_face_feats_path = os.path.join(self.face_root, FEAT_PATH, FACE_TRAIN_NAME)
            self.val_face_feats_path = os.path.join(self.face_root, FEAT_PATH, FACE_VAL_NAME)
            self.train_scene_feats_path = os.path.join(self.scene_root, SCENE_TRAIN_NAME)
            self.val_scene_feats_path = os.path.join(self.scene_root, SCENE_VAL_NAME)
            self.train_gt_path = os.path.join(self.face_root, TRAIN_GT_NAME)
            self.val_gt_path = os.path.join(self.face_root, VAL_GT_NAME)
        elif self.tvt == 'test':
            self.face_feats_path = os.path.join(self.face_root, FEAT_PATH, FACE_TEST_NAME)
            self.scene_feats_path = os.path.join(self.scene_root, SCENE_TEST_NAME)
            self.gt_path = None

        self._init_feat_labels()

    def _init_feat_labels(self):
        if self.tvt == 'train':
            face_feat_info = load_face_from_pickle(self.face_feats_path)
            scene_feat_info = load_scene_infos(self.scene_feats_path)
            gt_labels = load_train_gt_from_txt(self.gt_path)
        elif self.tvt == 'val':
            face_feat_info = load_face_from_pickle(self.face_feats_path)
            scene_feat_info = load_scene_infos(self.scene_feats_path)
            gt_labels = load_val_gt_from_txt(self.gt_path)
        elif self.tvt == 'train+val' or self.tvt == 'train+val-noise':
            face_feat_info = []
            face_feat_info += load_face_from_pickle(self.train_face_feats_path)
            face_feat_info += load_face_from_pickle(self.val_face_feats_path)

            scene_feat_info = {}
            scene_feat_info.update(load_scene_infos(self.train_scene_feats_path))
            scene_feat_info.update(load_scene_infos(self.val_scene_feats_path))

            gt_labels = {}
            gt_labels.update(load_train_gt_from_txt(self.train_gt_path))
            gt_labels.update(load_val_gt_from_txt(self.val_gt_path))
        else:
            face_feat_info = load_face_from_pickle(self.face_feats_path)
            scene_feat_info = load_scene_infos(self.scene_feats_path)
            gt_labels = {}

        self.vid_infos = self.pre_progress(face_feat_info, scene_feat_info, gt_labels, **self.kwargs)
        if self.tvt == 'train+val-noise':
            self.vid_infos = default_face_scene_remove_noise_in_val(self.vid_infos, **self.kwargs)
        self.length = len(self.vid_infos)

    def __getitem__(self, index):
        vid_info = self.vid_infos[index]
        label = vid_info['label']
        video_name = vid_info['video_name']

        face_feat, scene_feat = self.transform(vid_info, **self.kwargs)
        label = self.target_transform(label, **self.kwargs)

        return face_feat, scene_feat, label, video_name

    def __len__(self):
        return self.length
