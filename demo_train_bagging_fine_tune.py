# -*- coding: utf-8 -*-
# @Time    : 2019/6/2 16:06
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_train_bagging_fine_tune.py
# @Software: PyCharm
import argparse
import math
import os
import random
import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from datasets import BaseDataLoader, IQiYiFineTuneSceneDataset
from models import FocalLoss, ArcMarginProduct, ArcFaceSEResNeXtModel
from utils import check_exists, topk_func, save_model, prepare_device, adjust_learning_rate


def run_train(epoch_idx, model, train_loader, optimizer, metric_func, loss_func, device, log_step):
    model.train()
    train_loader.dataset.set_val(False)
    print(train_loader.dataset.is_val)

    total_loss = .0
    start = time.time()

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.view(-1, *images.size()[-3:])
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs.view(outputs.size(0) // 3, 3, -1)
        outputs = torch.mean(outputs, dim=1)
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
    val_loader.dataset.set_val(True)
    print(val_loader.dataset.is_val)

    total_top1 = .0
    total_top5 = .0
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(tqdm(val_loader, desc='Val Model')):
            images = images.view(-1, *images.size()[-3:])
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0) // 3, 3, -1)
            outputs = torch.mean(outputs, dim=1)
            total_top1 += val_func(outputs, labels, 1)
            total_top5 += val_func(outputs, labels, 5)

    test_log = {'top1 acc': 100. * total_top1 / len(val_loader),
                'top5 acc': 100. * total_top5 / len(val_loader), }

    return test_log


def main(args):
    if not check_exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = IQiYiFineTuneSceneDataset(args.data_root, 'train+val-noise', image_root='/home/dcq/img')
    train_loader = BaseDataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  validation_split=0.1, num_workers=16)
    val_loader = train_loader.split_validation()

    train_log_step = len(train_loader) // 10 if len(train_loader) > 10 else 1

    model = ArcFaceSEResNeXtModel(args.num_classes, )
    metric_func = ArcMarginProduct()
    loss_func = FocalLoss()

    trainable_params = [
        {'params': model.base_model.parameters(), "lr": args.learning_rate / 100},
        {'params': model.fc.parameters()},
        {'params': model.weight},
    ]

    warm_up_epoch = math.ceil(0.1 * args.epoch)
    optimizer = optim.SGD(trainable_params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch - warm_up_epoch)

    device, device_ids = prepare_device()
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    max_test_acc = .0
    for epoch_idx in range(args.epoch):
        if epoch_idx <= warm_up_epoch:
            adjust_learning_rate(optimizer, epoch_idx, warm_up_epoch, args.learning_rate)
        if epoch_idx >= warm_up_epoch:
            lr_scheduler.step()

        train_log = run_train(epoch_idx, model, train_loader, optimizer, metric_func, loss_func, device, train_log_step)
        for key, value in sorted(train_log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        test_log = run_val(model, val_loader, topk_func, device)
        for key, value in sorted(test_log.items(), key=lambda item: item[0]):
            print('    {:20s}: {:6f}'.format(str(key), value))

        # if epoch_idx % args.save_interval == 0 and epoch_idx != 0:
        #     save_model(model, args.save_dir, 'test_model', epoch_idx, is_best=False)

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
                        help='path to save model (default: ./checkpoints//bagging/)')
    parser.add_argument('--epoch', type=int, default=30, help="the epoch num for train (default: 30)")
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--num_classes', default=10035, type=int, help='number of classes (default: 10035)')
    parser.add_argument('--batch_size', default=40, type=int, help='dim of feature (default: 40)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help="learning rate for model (default: 0.1)")
    parser.add_argument('--save_interval', type=int, default=5, help='save interval in train model (default: 5)')

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
