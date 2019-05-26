#!/usr/bin/env bash
python -u demo_test_perceptron.py --device 2 --num_attn 1 --middle_ratio 2 --block_num 1
python -u demo_test_perceptron.py --device 2 --num_attn 1 --middle_ratio 2 --block_num 3
python -u demo_test_perceptron.py --device 2 --num_attn 1 --middle_ratio 4 --block_num 4
