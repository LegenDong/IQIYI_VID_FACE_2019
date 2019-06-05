#!/usr/bin/env bash

python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug10_mixup_0.1/ --aug_mixup_rate 0.1 --aug_num_vid 10
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug10_mixup_0.2/ --aug_mixup_rate 0.2 --aug_num_vid 10
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug10_mixup_0.3/ --aug_mixup_rate 0.3 --aug_num_vid 10
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug10_mixup_0.4/ --aug_mixup_rate 0.4 --aug_num_vid 10
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug10_mixup_0.5/ --aug_mixup_rate 0.5 --aug_num_vid 10

python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug20_mixup_0.1/ --aug_mixup_rate 0.1 --aug_num_vid 20
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug20_mixup_0.2/ --aug_mixup_rate 0.2 --aug_num_vid 20
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug20_mixup_0.3/ --aug_mixup_rate 0.3 --aug_num_vid 20
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug20_mixup_0.4/ --aug_mixup_rate 0.4 --aug_num_vid 20
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug20_mixup_0.5/ --aug_mixup_rate 0.5 --aug_num_vid 20

python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug30_mixup_0.1/ --aug_mixup_rate 0.1 --aug_num_vid 30
python -u demo_train_nan_aug_mixup.py --device 2 --save_dir ./checkpoints/nan_aug30_mixup_0.2/ --aug_mixup_rate 0.2 --aug_num_vid 30