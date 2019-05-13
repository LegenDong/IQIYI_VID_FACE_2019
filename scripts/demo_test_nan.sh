#!/usr/bin/env bash

python -u demo_test_nan.py --device 2 --load_path ./checkpoint/2_10/test_model_0079.pth --log_root ./logs/2_10/ --result_root ./result/2_10/ --num_attn 1
python evaluation_map.py --result_root ./result/2_10/result.txt

python -u demo_test_nan.py --device 2 --load_path ./checkpoint/2_20/test_model_0079.pth --log_root ./logs/2_20/ --result_root ./result/2_20/ --num_attn 1
python evaluation_map.py --result_root ./result/2_20/result.txt

python -u demo_test_nan.py --device 2 --load_path ./checkpoint/2_30/test_model_0079.pth --log_root ./logs/2_30/ --result_root ./result/2_30/ --num_attn 1
python evaluation_map.py --result_root ./result/2_30/result.txt

python -u demo_test_nan.py --device 2 --load_path ./checkpoint/2_40/test_model_0079.pth --log_root ./logs/2_40/ --result_root ./result/2_40/ --num_attn 1
python evaluation_map.py --result_root ./result/2_40/result.txt

python -u demo_test_nan.py --device 2 --load_path ./checkpoint/2_50/test_model_0079.pth --log_root ./logs/2_50/ --result_root ./result/2_50/ --num_attn 1
python evaluation_map.py --result_root ./result/2_50/result.txt
