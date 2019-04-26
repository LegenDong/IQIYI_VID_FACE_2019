# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 20:36
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_train_arcface.py
# @Software: PyCharm
import argparse
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from datasets import IQiYiFaceDataset
from models import ArcFaceModel, FocalLoss, ArcMarginProduct
from utils import check_exists, save_model, weighted_average_pre_progress


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = IQiYiFaceDataset(args.root, 'train', pre_progress=weighted_average_pre_progress)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    log_step = len(data_loader) // 10 if len(data_loader) > 10 else 1

    model = ArcFaceModel(args.feat_dim, args.num_classes)
    metric_func = ArcMarginProduct()
    loss_func = FocalLoss()

    optimizer_model = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer_model, 40)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch_idx in range(200):
        total_loss = .0
        for batch_idx, (feats, labels, _) in enumerate(data_loader):

            feats = feats.to(device)
            labels = labels.to(device)

            optimizer_model.zero_grad()

            outputs = model(feats)
            outputs_metric = metric_func(outputs, labels)
            local_loss = loss_func(outputs_metric, labels)

            local_loss.backward()
            optimizer_model.step()

            total_loss += local_loss.item()

            if batch_idx % log_step == 0 and batch_idx != 0:
                print('Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'
                      .format(epoch_idx, batch_idx * args.batch_size, len(dataset),
                              100.0 * batch_idx / len(data_loader), local_loss.item()))
        log = {'epoch': epoch_idx,
               'lr': optimizer_model.param_groups[0]['lr'],
               'loss': total_loss / len(data_loader)}

        for key, value in sorted(log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        lr_scheduler.step()

        save_model(model, args.save_dir, 'test_model', epoch_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--root', default='/data/dcq/DataSets/iQIYI/', type=str,
                        help='path to load data (default: /data/dcq/DataSets/iQIYI/)')
    parser.add_argument('-s', '--save_dir', default='/data/dcq/Models/iQIYI/', type=str,
                        help='path to save model (default: /data/dcq/Models/iQIYI/)')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('-n', '--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('-b', '--batch_size', default=4096, type=int, help='dim of feature (default: 4096)')
    parser.add_argument('-dim', '--feat_dim', default=512, type=int, help='dim of feature (default: 512)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help="learning rate for model (default: 0.1)")
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    main(args)
