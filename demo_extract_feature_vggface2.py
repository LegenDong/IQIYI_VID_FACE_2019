# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 17:10
# @Author  : LegenDong
# @User    : legendong
# @File    : demo_extract_feature_vggface2.py
# @Software: PyCharm
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import IQiYiFaceImageDataset
from models import VGGFaceModel
from utils import load_face_from_pickle


def main():
    dataset = IQiYiFaceImageDataset('/data/materials/', tvt='val')
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    model = VGGFaceModel(num_classes=10035, include_top=False)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    video_infos = {}
    start = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels, video_names) in enumerate(data_loader):
            images = images.to(device)
            output = model(images)
            output_np = output.cpu().numpy().astype(np.float16)
            for idx, video_name in enumerate(video_names):
                video_infos.setdefault(video_name.encode('utf-8'), []) \
                    .append(('0', (0., 0., 0., 0.), 1.0, 200., output_np[idx]))

            end = time.time()
            print(end - start)
            start = time.time()

    with open('./vggface2/feat/face_val_v2.pickle', 'wb') as fout:
        pickle.dump(video_infos, fout)


if __name__ == '__main__':
    main()

    video_infos = load_face_from_pickle('./vggface2/feat/face_val_v2.pickle')
    flag = 10
    for video_info in video_infos:
        print(len(video_info['frame_infos']))
        print(video_info['video_name'])
        flag -= 1
        if flag == 0:
            break
