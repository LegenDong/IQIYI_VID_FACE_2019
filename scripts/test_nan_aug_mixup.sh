#!/usr/bin/env bash

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.1 --device 1 --result_root ./result/nan_aug20_mixup_0.1/ --log_root ./logs/nan_aug20_mixup_0.1
python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.1/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.2 --device 1 --result_root ./result/nan_aug20_mixup_0.2/ --log_root ./logs/nan_aug20_mixup_0.2
python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.2/result.txt

#python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.3 --device 1 --result_root ./result/nan_aug20_mixup_0.3/ --log_root ./logs/nan_aug20_mixup_0.3
#python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.3/result.txt
#
#python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.4 --device 1 --result_root ./result/nan_aug20_mixup_0.4/ --log_root ./logs/nan_aug20_mixup_0.4
#python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.4/result.txt
#
#python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.5 --device 1 --result_root ./result/nan_aug20_mixup_0.5/ --log_root ./logs/nan_aug20_mixup_0.5
#python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.5/result.txt