#!/usr/bin/env bash
python -u demo_train_perceptron.py --device 0 --num_attn 1 --middle_ratio 4 --block_num 4 --drop_prob .60 --prelu_init .15 --seed 0
python -u demo_train_perceptron.py --device 0 --num_attn 1 --middle_ratio 3 --block_num 4 --drop_prob .55 --prelu_init .25 --seed 1
python -u demo_train_perceptron.py --device 0 --num_attn 1 --middle_ratio 3 --block_num 3 --drop_prob .45 --prelu_init .35 --seed 2
