# -*- coding: utf-8 -*-
# @Time    : 2019-05-30 13:08
# @Author  : edward
# @File    : main_noise.py
# @Software: PyCharm
from datasets import IQiYiVidDataset

if __name__ == '__main__':
    pred_root = './result/nan_aug10_mixup_0.4/result.txt'
    gt_path = '/data/materials/val_gt.txt'

    gt_id2vid = dict()
    with open(gt_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split(' ')
            gt_id2vid[terms[0]] = terms[1:]
    fin.close()
    id_num = len(lines)

    gt_vid2id = {}
    for key, values in gt_id2vid.items():
        for value in values:
            gt_vid2id.setdefault(value, []).append(key)

    pred_id2vid = dict()
    with open(pred_root, 'r') as fin:
        lines = fin.readlines()
        assert (len(lines) <= id_num)
        for line in lines:
            terms = line.strip().split(' ')
            tmp_list = []
            for video in terms[1:]:
                if video not in tmp_list:
                    tmp_list.append(video)
            pred_id2vid[terms[0]] = tmp_list
    fin.close()

    pred_vid2id = {}
    for key, values in pred_id2vid.items():
        for value in values:
            pred_vid2id.setdefault(value, []).append(key)

    gt_vid = set(gt_vid2id.keys())
    pred_vid = set(pred_vid2id.keys())
    # print(len(gt_vid))
    # print(len(pred_vid))

    noise_vid_names = set()

    dataset = IQiYiVidDataset('/data/materials', 'val', 'face')
    # print(len(dataset))
    for info in dataset.vid_infos:
        video_name = info['video_name']
        if video_name + '.mp4' not in gt_vid2id.keys():
            noise_vid_names.add(video_name + '.mp4')
        label = info['label']
        qs = [frame['quality_score'] for frame in info['face']]

    print(len(noise_vid_names))

    pred_noises = {}
    count = 0
    ap = []
    ap_total = 0.
    with open('ana.txt', 'w', encoding='utf-8') as f:
        for cid in gt_id2vid:
            videos = gt_id2vid[cid]
            if cid not in pred_id2vid:
                continue
            my_videos = pred_id2vid[cid]
            # recall number upper bound
            assert (len(my_videos) <= 100)
            ap = 0.
            ind = 0.
            for ind_video, my_video in enumerate(my_videos):
                if my_video in videos:
                    ind += 1
                    ap += ind / (ind_video + 1)
            if ap / len(videos) < 0.3:
                ap_total += 0.3
            else:
                ap_total += ap / len(videos)
            f.write(str(cid) + ' ' + str(ap / len(videos)) + '\n')
    print(ap_total / 10034)
    #     for key, values in gt_id2vid.items():
    #         state = []
    #         ap = 0.
    #         ind = 0.
    #         for ind_video, value in enumerate(values):
    #             if value in noise_vid_names:
    #                 state.append(0)
    #             elif value in gt_id2vid[key]:
    #                 ind += 1
    #                 ap += ind / (ind_video + 1)
    #                 state.append(1)
    #             else:
    #                 state.append(2)
    #         # ap_total += ap / len(values)
    #         f.write(str(key) + ' ' + str(ap / len(values)) + '\n')
    # f.close()
    line_map_info = []
    with open('ana.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line_idx, line in enumerate(lines):
            split = line.strip().split(' ')
            line_map = float(split[1])
            if line_map < 0.30:
                print(line_idx + 1)
                line_map_info.append('{}: {} '.format(line_idx + 1, line_map))

    # print(line_map_info)

