#!/usr/bin/env bash
python -u demo_extract_scene.py --tvt test
python -u demo_test_face_scene_multi_view.py --seed 0 --device 0
python -u demo_test_face_scene_multi_view.py --seed 1 --device 0
python -u demo_test_face_scene_multi_view.py --seed 2 --device 0
python -u demo_test_face_scene_multi_view.py --seed 4 --device 0
python -u demo_test_face_scene_multi_view.py --seed 5 --device 0
python -u demo_test_face_scene_multi_view.py --seed 6 --device 0
python -u merge_multi_files.py --merge_type face_scene
python -u demo_test_face_scene_multi_view.py --seed 8 --device 0
python -u demo_test_face_scene_multi_view.py --seed 9 --device 0
python -u demo_test_face_scene_multi_view.py --seed 10 --device 0
python -u demo_test_face_scene_multi_view.py --seed 12 --device 0
python -u demo_test_face_scene_multi_view.py --seed 15 --device 0
python -u merge_multi_files.py --merge_type face_scene
python -u demo_test_scene_multi_view.py --seed 1 --device 0
python -u demo_test_scene_multi_view.py --seed 2 --device 0
python -u demo_test_scene_multi_view.py --seed 3 --device 0
python -u demo_test_scene_multi_view.py --seed 4 --device 0
python -u demo_test_scene_multi_view.py --seed 6 --device 0
python -u merge_multi_files.py --merge_type scene
python -u demo_test_scene_multi_view.py --seed 8 --device 0
python -u demo_test_scene_multi_view.py --seed 10 --device 0
python -u demo_test_scene_multi_view.py --seed 11 --device 0
python -u demo_test_scene_multi_view.py --seed 12 --device 0
python -u demo_test_scene_multi_view.py --seed 14 --device 0
python -u demo_test_scene_multi_view.py --seed 15 --device 0
python -u merge_multi_files.py --merge_type scene
python -u main.py
