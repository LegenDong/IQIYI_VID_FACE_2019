# -*- coding: utf-8 -*-
# @Time    : 2019/5/31 16:10
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_train_multi_view_maxout.py
# @Software: PyCharm
import argparse
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from datasets import IQiYiVidDataset
from models import FocalLoss, ArcMarginProduct, ArcFaceNanMaxOutModel
from utils import check_exists, save_model, sep_cat_qds_select_vid_transforms, get_mask_index


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_id = args.seed

    assert args.moda in ['face', 'head']
    mask_index = get_mask_index(args.seed, 512, 16)
    print(mask_index)

    dataset = IQiYiVidDataset(args.data_root, 'train+val', args.moda, transform=sep_cat_qds_select_vid_transforms,
                              mask_index=mask_index, num_frame=args.num_frame, )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    log_step = len(data_loader) // 10 if len(data_loader) > 10 else 1

    model = ArcFaceNanMaxOutModel(args.feat_dim, args.num_classes, num_attn=args.num_attn,
                                  stuff_labels=args.stuff_labels)
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

    save_model(model, args.save_dir, 'demo_arcface_{}_multi_view_stuff_{}_{}_model'
               .format(args.moda, args.stuff_labels, model_id), args.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--data_root', default='/data/materials', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--save_dir', default='./checkpoints/multi_view', type=str,
                        help='path to save model (default: ./checkpoints/multi_view)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=4096, type=int, help='dim of feature (default: 4096)')
    parser.add_argument('--feat_dim', default=480 + 2, type=int, help='dim of feature (default: 480 + 2)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate for model (default: 0.1)")
    parser.add_argument('--num_frame', default=40, type=int, help='size of video length (default: 40)')
    parser.add_argument('--num_attn', default=1, type=int, help='number of attention block in NAN')
    parser.add_argument('--moda', default='face', type=str, help='modal[face, head] of model train, (default: face)')
    parser.add_argument('--seed', default=0, type=int, help='seed for all random module (default: 0)')
    parser.add_argument('--stuff_labels', default=1000, type=int, help='stuff num for maxout model (default: 1000)')

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
