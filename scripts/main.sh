#!/usr/bin/env bash
#python -u demo_extract_scene.py --tvt test
#python -u demo_test_scene_feat.py
python -u demo_test_multi_view.py --seed 0 --device 0
python -u demo_test_multi_view.py --seed 1 --device 0
python -u demo_test_multi_view.py --seed 2 --device 0
python -u demo_test_multi_view.py --seed 3 --device 0
python -u demo_test_multi_view.py --seed 4 --device 0
python -u demo_test_multi_view.py --seed 5 --device 0
python -u demo_test_multi_view.py --seed 6 --device 0
python -u demo_test_multi_view.py --seed 7 --device 0
python -u merge_multi_files.py
python -u demo_test_multi_view.py --seed 8 --device 0
python -u demo_test_multi_view.py --seed 9 --device 0
python -u demo_test_multi_view.py --seed 10 --device 0
python -u demo_test_multi_view.py --seed 11 --device 0
python -u demo_test_multi_view.py --seed 12 --device 0
python -u demo_test_multi_view.py --seed 13 --device 0
python -u demo_test_multi_view.py --seed 14 --device 0
python -u demo_test_multi_view.py --seed 15 --device 0
python -u merge_multi_files.py
python -u demo_test_multi_view.py --seed 16 --device 0
python -u demo_test_multi_view.py --seed 17 --device 0
python -u demo_test_multi_view.py --seed 18 --device 0
python -u demo_test_multi_view.py --seed 19 --device 0
python -u demo_test_multi_view.py --seed 20 --device 0
python -u demo_test_multi_view.py --seed 21 --device 0
python -u demo_test_multi_view.py --seed 22 --device 0
python -u demo_test_multi_view.py --seed 23 --device 0
python -u merge_multi_files.py
python -u demo_test_multi_view.py --seed 24 --device 0
python -u demo_test_multi_view.py --seed 25 --device 0
python -u demo_test_multi_view.py --seed 26 --device 0
python -u demo_test_multi_view.py --seed 27 --device 0
python -u demo_test_multi_view.py --seed 28 --device 0
python -u demo_test_multi_view.py --seed 29 --device 0
python -u demo_test_multi_view.py --seed 30 --device 0
python -u demo_test_multi_view.py --seed 31 --device 0
python -u merge_multi_files.py
python -u main.py