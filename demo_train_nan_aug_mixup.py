# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 17:21
# @Author  : edward
# @File    : demo_train_nan.py
# @Software: PyCharm
import argparse
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from datasets import IQiYiVidDataset
from models import ArcFaceNanModel, FocalLoss, ArcMarginProduct
from utils import check_exists, save_model, aug_vid_pre_progress, sep_cat_qds_mixup_vid_transforms


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    assert args.moda in ['face', 'head']

    dataset = IQiYiVidDataset(args.data_root, 'train+val-noise', args.moda,
                              transform=sep_cat_qds_mixup_vid_transforms,
                              num_frame=args.num_frame,
                              pre_progress=aug_vid_pre_progress,
                              aug_num_vid=args.aug_num_vid,
                              aug_mixup_rate=args.aug_mixup_rate, )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    log_step = len(data_loader) // 10 if len(data_loader) > 10 else 1

    model = ArcFaceNanModel(args.feat_dim, args.num_classes, num_attn=args.num_attn)
    metric_func = ArcMarginProduct()
    loss_func = FocalLoss(gamma=args.focal_gamma)

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

    save_model(model, args.save_dir, 'demo_arcface_{}_nan_model'.format(args.moda), args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='/data/materials', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--save_dir', default='./checkpoints/', type=str,
                        help='path to save model (default: ./checkpoints/)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=4096, type=int, help='dim of feature (default: 4096)')
    parser.add_argument('--feat_dim', default=512 + 2, type=int, help='dim of feature (default: 512 + 2)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate for model (default: 0.1)")
    parser.add_argument('--num_frame', default=40, type=int, help='size of video length (default: 40)')
    parser.add_argument('--num_attn', default=1, type=int, help='number of attention block in NAN')
    parser.add_argument('--moda', default='face', type=str, help='modal[face, head] of model train, (default: face)')
    parser.add_argument('--focal_gamma', default=2., type=float, help='gamma for focal loss')
    parser.add_argument('--aug_mixup_rate', default=0.4, type=float, help='mix up rate for aug')
    parser.add_argument('--aug_num_vid', default=10, type=int, help='mix up rate for aug')

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
