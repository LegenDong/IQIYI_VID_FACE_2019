#!/usr/bin/env bash
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/1_10/ --embedding_size 10 --num_epoch 80 --num_decay 20 --num_attn 1
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/1_20/ --embedding_size 20 --num_epoch 80 --num_decay 20 --num_attn 1
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/1_30/ --embedding_size 30 --num_epoch 80 --num_decay 20 --num_attn 1
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/1_40/ --embedding_size 40 --num_epoch 80 --num_decay 20 --num_attn 1
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/1_50/ --embedding_size 50 --num_epoch 80 --num_decay 20 --num_attn 1