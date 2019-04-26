# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:36
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_train.py
# @Software: PyCharm
import argparse
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from datasets import IQiYiFaceDataset
from models import TestModel, TestSplitModel, FocalLoss, CenterLoss, RingLoss
from utils import check_exists, save_model, weighted_average_pre_progress


def main():
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = IQiYiFaceDataset(args.root, 'train',
                               pre_progress=weighted_average_pre_progress)
    data_loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=4)

    log_step = len(data_loader) // 10 if len(data_loader) > 10 else 1

    model = TestSplitModel()
    focal_loss = FocalLoss()
    # center_loss = CenterLoss(num_classes=args.num_classes, feat_dim=args.feat_dim)
    # ring_loss = RingLoss()

    optimizer_model = optim.SGD(model.parameters(), lr=args.lr_model, momentum=0.9, weight_decay=1e-5)
    # optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=args.lr_cent)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_model, 20)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch_idx in range(200):
        total_loss = .0
        for batch_idx, (feats_in, labels, _) in enumerate(data_loader):

            feats_in = feats_in.to(device)
            labels = labels.to(device)

            optimizer_model.zero_grad()
            # optimizer_centloss.zero_grad()

            feats_out, pred = model(feats_in, labels)

            loss_focal = focal_loss(pred, labels)
            # loss_center = args.weight_cent * center_loss(feats_out, labels)
            # loss_ring = args.weight_ring * ring_loss(pred)

            loss = loss_focal
            loss.backward()

            optimizer_model.step()
            # for param in center_loss.parameters():
            #     param.grad.data *= (1. / args.weight_cent)
            # optimizer_centloss.step()

            total_loss += loss.item()

            if batch_idx % log_step == 0 and batch_idx != 0:
                print('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'
                      .format(epoch_idx, batch_idx * 4096, len(dataset),
                              100.0 * batch_idx / len(data_loader), loss.item()))
        log = {'epoch': epoch_idx,
               'lr': optimizer_model.param_groups[0]['lr'],
               'loss': total_loss / len(data_loader)}

        for key, value in sorted(log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        lr_scheduler.step()

        save_model(model, args.save_dir, 'test_model', epoch_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--root', default='/data/gz/datasets/iQIYI/', type=str,
                        help='path to load data (default: /data/gz/datasets/iQIYI/)')
    parser.add_argument('-s', '--save_dir', default='/data/gz/models/iQIYI/default', type=str,
                        help='path to save model (default: /data/gz/models/iQIYI/default)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--num_classes', default=10035, type=int,
                        help='number of classes')
    parser.add_argument('-dim', '--feat_dim', default=512, type=int,
                        help='dim of feature')
    parser.add_argument('--lr-model', type=float, default=0.1, help="learning rate for model")
    parser.add_argument('--weight-cent', type=float, default=0., help="weight for center loss")
    parser.add_argument('--weight-ring', type=float, default=1., help="weight for ring loss")
    parser.add_argument('--lr-cent', type=float, default=0.1, help="learning rate for center loss")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    main()
