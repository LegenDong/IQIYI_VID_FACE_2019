#!/usr/bin/env bash
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/2_10/ --embedding_size 10 --num_epoch 200 --num_decay 40 --num_attn 2
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/2_20/ --embedding_size 20 --num_epoch 200 --num_decay 40 --num_attn 2
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/2_30/ --embedding_size 30 --num_epoch 200 --num_decay 40 --num_attn 2
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/2_40/ --embedding_size 40 --num_epoch 200 --num_decay 40 --num_attn 2
python -u demo_train_nan.py --device 3 --save_dir ./checkpoint/2_50/ --embedding_size 50 --num_epoch 200 --num_decay 40 --num_attn 2