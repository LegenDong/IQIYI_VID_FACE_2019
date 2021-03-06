# -*- coding: utf-8 -*-
# @Time    : 2019/6/3 12:17
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_train_scene_multi_view.py
# @Software: PyCharm
import argparse
import os
import pickle
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from datasets import IQiYiSceneFeatDataset
from models import FocalLoss, ArcMarginProduct, ArcSceneFeatModel
from utils import check_exists, save_model, get_mask_index, default_sep_select_scene_feat_transforms


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_id = args.seed

    mask_index = get_mask_index(args.seed, args.feat_dim, 32)
    print(mask_index)

    with open(os.path.join(args.save_dir, 'scene_mask_index_file_{}.pickle'.format(model_id)), 'wb') as fout:
        pickle.dump(mask_index, fout)

    dataset = IQiYiSceneFeatDataset(args.data_root, 'train+val-noise', mask_index=mask_index,
                                    transform=default_sep_select_scene_feat_transforms)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    log_step = len(data_loader) // 10 if len(data_loader) > 10 else 1

    model = ArcSceneFeatModel(len(mask_index), args.num_classes, )
    metric_func = ArcMarginProduct()
    loss_func = FocalLoss(gamma=2.)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch_idx in range(args.epoch):
        total_loss = .0
        for batch_idx, (feats, labels, _) in enumerate(data_loader):
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(feats)
            outputs_metric = metric_func(outputs, labels)
            local_loss = loss_func(outputs_metric, labels)

            local_loss.backward()
            optimizer.step()

            total_loss += local_loss.item()

            if batch_idx % log_step == 0 and batch_idx != 0:
                print('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'
                      .format(epoch_idx, batch_idx * args.batch_size, len(dataset),
                              100.0 * batch_idx / len(data_loader), local_loss.item()))
        log = {'epoch': epoch_idx,
               'lr': optimizer.param_groups[0]['lr'],
               'loss': total_loss / len(data_loader)}

        for key, value in sorted(log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        lr_scheduler.step()

    save_model(model, args.save_dir, 'demo_arcface_scene_multi_view_{}_model'.format(model_id), args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='./scene_feat/', type=str,
                        help='path to load data (default: ./scene_feat/)')
    parser.add_argument('--save_dir', default='./checkpoints/multi_view_scene', type=str,
                        help='path to save model (default: ./checkpoints/multi_view_scene)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=4096, type=int, help='dim of feature (default: 4096)')
    parser.add_argument('--feat_dim', default=2048, type=int, help='dim of feature (default: 2048)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate for model (default: 0.1)")
    parser.add_argument('--seed', default=0, type=int, help='seed for all random module (default: 0)')

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    main(args)
