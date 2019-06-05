#!/usr/bin/env bash

python -u demo_test_nan.py --load_root ./checkpoints/aug_mixup_30 --device 3 --result_root ./result/aug_mixup_30/ --log_root ./logs/aug_mixup_30
python evaluation_map.py --result_root ./result/aug_mixup_30/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/aug_mixup_40 --device 3 --result_root ./result/aug_mixup_40/ --log_root ./logs/aug_mixup_40
python evaluation_map.py --result_root ./result/aug_mixup_40/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/aug_mixup_50 --device 3 --result_root ./result/aug_mixup_50/ --log_root ./logs/aug_mixup_50
python evaluation_map.py --result_root ./result/aug_mixup_50/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.3 --device 3 --result_root ./result/nan_aug20_mixup_0.3/ --log_root ./logs/nan_aug20_mixup_0.3
python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.3/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.4 --device 3 --result_root ./result/nan_aug20_mixup_0.4/ --log_root ./logs/nan_aug20_mixup_0.4
python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.4/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug20_mixup_0.5 --device 3 --result_root ./result/nan_aug20_mixup_0.5/ --log_root ./logs/nan_aug20_mixup_0.5
python evaluation_map.py --result_root ./result/nan_aug20_mixup_0.5/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug30_mixup_0.1 --device 3 --result_root ./result/nan_aug30_mixup_0.1/ --log_root ./logs/nan_aug30_mixup_0.1
python evaluation_map.py --result_root ./result/nan_aug30_mixup_0.1/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug30_mixup_0.2 --device 3 --result_root ./result/nan_aug30_mixup_0.2/ --log_root ./logs/nan_aug30_mixup_0.2
python evaluation_map.py --result_root ./result/nan_aug30_mixup_0.2/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug30_mixup_0.3 --device 3 --result_root ./result/nan_aug30_mixup_0.3/ --log_root ./logs/nan_aug30_mixup_0.3
python evaluation_map.py --result_root ./result/nan_aug30_mixup_0.3/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug30_mixup_0.4 --device 3 --result_root ./result/nan_aug30_mixup_0.4/ --log_root ./logs/nan_aug30_mixup_0.4
python evaluation_map.py --result_root ./result/nan_aug30_mixup_0.4/result.txt

python -u demo_test_nan.py --load_root ./checkpoints/nan_aug30_mixup_0.5 --device 3 --result_root ./result/nan_aug30_mixup_0.5/ --log_root ./logs/nan_aug30_mixup_0.5
python evaluation_map.py --result_root ./result/nan_aug30_mixup_0.5/result.txt
