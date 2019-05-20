# -*- coding: utf-8 -*-
# @Time    : 2019/5/11 15:09
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_train_bagging_mm.py
# @Software: PyCharm
import argparse
import os
import random
import time

import numpy as np
import torch
from torch import optim

from datasets import BaseDataLoader, IQiYiVidDataset
from models import FocalLoss, ArcMarginProduct, ArcFaceMultiModalModel
from utils import check_exists, topk_func, save_model


def run_train(epoch_idx, model, train_loader, optimizer, metric_func, loss_func, device, log_step):
    model.train()

    total_loss = .0
    start = time.time()

    for batch_idx, (feats1, feats2, labels, _) in enumerate(train_loader):
        feats1 = feats1.to(device)
        feats2 = feats2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(feats1, feats2)
        outputs_metric = metric_func(outputs, labels)
        local_loss = loss_func(outputs_metric, labels)

        local_loss.backward()
        optimizer.step()

        total_loss += local_loss.item()

        if batch_idx % log_step == 0 and batch_idx != 0:
            end = time.time()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Time: {:.2f}'
                  .format(epoch_idx, batch_idx * args.batch_size, train_loader.n_samples,
                          100.0 * batch_idx / len(train_loader), local_loss.item(), end - start))

            start = time.time()

    train_log = {'epoch': epoch_idx,
                 'lr': optimizer.param_groups[0]['lr'],
                 'loss': total_loss / len(train_loader)}

    return train_log


def run_val(model, val_loader, val_func, device):
    model.eval()

    total_top1 = .0
    total_top5 = .0
    with torch.no_grad():
        for batch_idx, (feats1, feats2, labels, _) in enumerate(val_loader):
            feats1 = feats1.to(device)
            feats2 = feats2.to(device)
            labels = labels.to(device)
            output = model(feats1, feats2)
            total_top1 += val_func(output, labels, 1)
            total_top5 += val_func(output, labels, 5)

    test_log = {'top1 acc': 100. * total_top1 / len(val_loader),
                'top5 acc': 100. * total_top5 / len(val_loader), }

    return test_log


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = IQiYiVidDataset(args.data_root, 'train+val-noise', modes='face+head')
    train_loader = BaseDataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  validation_split=0.1, num_workers=4)
    val_loader = train_loader.split_validation()

    train_log_step = len(train_loader) // 10 if len(train_loader) > 10 else 1

    model = ArcFaceMultiModalModel(args.feat_dim, args.num_classes)
    metric_func = ArcMarginProduct()
    loss_func = FocalLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    max_test_acc = .0
    for epoch_idx in range(args.epoch):
        lr_scheduler.step()

        train_log = run_train(epoch_idx, model, train_loader, optimizer, metric_func, loss_func, device, train_log_step)
        for key, value in sorted(train_log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        test_log = run_val(model, val_loader, topk_func, device)
        for key, value in sorted(test_log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        if epoch_idx % args.save_interval == 0 and epoch_idx != 0:
            save_model(model, args.save_dir, 'test_model', epoch_idx, is_best=False)

        test_acc = test_log['top1 acc']
        if max_test_acc < test_acc:
            save_model(model, args.save_dir, 'best_model', epoch_idx, is_best=True)
            max_test_acc = test_acc

        local_logs = train_log.copy()
        local_logs.update(test_log)
        with open(os.path.join(args.save_dir, 'logs.txt'), 'a', encoding='utf-8') as f:
            if epoch_idx == 0:
                log_str = '\t'.join([str(key) for key, _ in sorted(local_logs.items(), key=lambda item: item[0])])
                f.write(log_str + '\n')

            log_str = ' '.join([str(value) for _, value in sorted(local_logs.items(), key=lambda item: item[0])])
            f.write(log_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IQIYI VID FACE 2019')
    parser.add_argument('--data_root', default='/data/materials', type=str,
                        help='path to load data (default: /data/materials/)')
    parser.add_argument('--save_dir', default='./checkpoints/bagging/', type=str,
                        help='path to save model (default: ./checkpoints/bagging/)')
    parser.add_argument('--epoch', type=int, default=100, help="the epoch num for train (default: 100)")
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=4096, type=int, help='dim of feature (default: 4096)')
    parser.add_argument('--feat_dim', default=512, type=int, help='dim of feature (default: 512)')
    parser.add_argument('--save_interval', default=10, type=int, help='interval of epochs for save model (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate for model (default: 0.1)")

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
