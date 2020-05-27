import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

from models import CPN
from data_gen import MscocoMulti
from torch.utils.data import DataLoader
from config import cfg
import os

checkpoint_path = 'BEST_checkpoint.tar'
anno_root = 'data/COCO2017/annotations/COCO_2017_val.json'


def get_model():
    model = CPN()
    model = torch.nn.DataParallel(model)
    if os.path.exists(checkpoint_path):
        print('=========load checkpoint========')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # for k, v in checkpoint['state_dict'].items():
        #     print(k, v.shape)
        model.load_state_dict(checkpoint['state_dict'])
    return model


# ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist",
#     "right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],
#	"skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
def get_kpts(maps, img_h=368.0, img_w=368.0):
    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x, y])
    return kpts


if __name__ == '__main__':
    model = get_model()
    data_set = MscocoMulti(cfg)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
    for i, (_, target, valid, mata) in enumerate(data_loader):
        print(i,'=====================')
        # global_pred,refine_pred=model(img)
        # print(mata)
        img_path = mata['img_path'][0]
        gt_box = mata['GT_bbox'][0]
        image = cv2.imread(img_path)
        # image=cv2.circle(image,center=(gt_box[0],gt_box[1]),radius=3,color=(0,255,0),thickness=2)
        # image = cv2.circle(image, center=(gt_box[2], gt_box[3]), radius=3, color=(0, 255, 0), thickness=2)
        image = image[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]]
        image = cv2.resize(image, (256, 192))
        image0 = image
        # print(image.shape)
        # cv2.imshow('', image)
        # cv2.waitKey(-1)

        image = image[:, :, ::-1]
        image = image / 255
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image).float().unsqueeze(0)
        # print(image.shape)
        global_pred, refine_pred = model(image)
        # print(refine_pred.shape)

        kpts = get_kpts(refine_pred, 192, 256)
        # kpts=get_kpts(target)
        # print(kpts)
        for t in kpts:
            image0 = cv2.circle(image0, center=(t[0], t[1]), radius=3, color=(0, 255, 0), thickness=2)
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                    [4, 6], [5, 7]]
        for line in skeleton:
            node0, node1 = line
            node0 -= 1
            node1 -= 1
            image0 = cv2.line(image0, (kpts[node0][0], kpts[node0][1]), (kpts[node1][0], kpts[node1][1]), color=(0, 0, 255), thickness=1)

        # cv2.imshow('image0', image0)
        cv2.imwrite('data/images/{0}.jpg'.format(i), image0)
        if i > 200:
            break
            # print(target)
