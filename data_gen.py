import os
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform
from PIL import Image
import imageio
import torch
import torch.utils.data as data
from utils.imutils import *
from utils.transforms import *


class MscocoMulti(data.Dataset):
    def __init__(self, cfg, train=True):
        self.img_folder = cfg.img_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.bbox_extend_factor = cfg.bbox_extend_factor
        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry
        with open(cfg.gt_path) as anno_file:
            self.anno = json.load(anno_file)

    def augmentationCropImage(self, img, bbox, joints=None):
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        mean_value = self.pixel_means
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add
        if self.is_train:
            joints[:, :2] += add
            inds = np.where(joints[:, -1] == 0)
            joints[inds, :2] = -1000000  # avoid influencing by data processing
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if self.is_train:
            crop_width = crop_width * (1 + 0.25)
            crop_height = crop_height * (1 + 0.25)
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        if self.is_train:
            joints[:, 0] = joints[:, 0] - min_x
            joints[:, 1] = joints[:, 1] - min_y

            joints[:, 0] *= x_ratio
            joints[:, 1] *= y_ratio
            label = joints[:, :2].copy()
            valid = joints[:, 2].copy()
        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

        if self.is_train:
            return img, joints, details
        else:
            return img, details

    def data_augmentation(self, img, label, operation):
        height, width = img.shape[0], img.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])

        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img = skimage.transform.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                                       int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        for i in range(n):
            label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            label[i][2] *= (
                (label[i][0] >= 0) & (label[i][0] < width) & (label[i][1] >= 0) & (label[i][1] < height))

        # flip augmentation
        if operation == 1:
            img = cv2.flip(img, 1)
            cod = []
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                if x >= 0:
                    x = width - 1 - x
                cod.append((x, y, label[i][2]))
            # **** the joint index depends on the dataset ****
            for (q, w) in self.symmetry:
                cod[q], cod[w] = cod[w], cod[q]
            for i in range(n):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
                allc.append(cod[i][2])
            label = np.array(allc).reshape(n, 3)

        # rotated augmentation
        if operation > 1:
            angle = random.uniform(0, self.rot_factor)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotMat, (width, height))

            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                v = label[i][2]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(int(coor[0]))
                allc.append(int(coor[1]))
                v *= ((coor[0] >= 0) & (coor[0] < width) & (coor[1] >= 0) & (coor[1] < height))
                allc.append(int(v))
            label = np.array(allc).reshape(n, 3).astype(np.int)
        return img, label

    def __getitem__(self, index):
        a = self.anno[index]
        # print(a)
        image_name = a['imgInfo']['img_paths']
        img_path = os.path.join(self.img_folder, image_name)
        # print(img_path)
        if self.is_train:
            points = np.array(a['unit']['keypoints']).reshape(self.num_class, 3).astype(np.float32)
        gt_bbox = a['unit']['GT_bbox']

        image = cv2.imread(img_path)
        image = image[:, :, ::-1]
        if self.is_train:
            image, points, details = self.augmentationCropImage(image, gt_bbox, points)
        else:
            image, details = self.augmentationCropImage(image, gt_bbox)

        if self.is_train:
            image, points = self.data_augmentation(image, points, random.randint(1, 2))
            img = im_to_torch(image)  # CxHxW

            # Color dithering
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

            points[:, :2] //= 4  # output size is 1/4 input size
            pts = torch.Tensor(points)
        else:
            img = im_to_torch(image)
        img = color_normalize(img, self.pixel_means)

        if self.is_train:
            target15 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target11 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target9 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target7 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            for i in range(self.num_class):
                if pts[i, 2] > 0:  # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                    target15[i] = generate_heatmap(target15[i], pts[i], self.cfg.gk15)
                    target11[i] = generate_heatmap(target11[i], pts[i], self.cfg.gk11)
                    target9[i] = generate_heatmap(target9[i], pts[i], self.cfg.gk9)
                    target7[i] = generate_heatmap(target7[i], pts[i], self.cfg.gk7)

            targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
            valid = pts[:, 2]

        meta = {'index': index, 'imgID': a['imgInfo']['imgID'],
                'GT_bbox': np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]),
                'img_path': img_path, 'augmentation_details': details}

        if self.is_train:
            return img, targets, valid, meta
        else:
            meta['det_scores'] = a['score']
            return img, meta

    def __len__(self):
        return len(self.anno)


if __name__ == '__main__':
    from config import cfg
    from torch.utils.data import DataLoader

    dataset = MscocoMulti(cfg, train=True)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for img, target, valid, mata in train_loader:
        print(len(train_loader))
        break


    # img3 = cv2.imread('data/COCO2017/val2017/000000000139.jpg')
    # img3 = img3[:, :, ::-1]
    # print(img3.shape)
    # print(img3[0][0])
    # img1 = imageio.imread('data/COCO2017/val2017/000000000139.jpg')
    # print(img1.shape)
    # print(img1[0][0])
    # image2 = np.array(Image.open('data/COCO2017/val2017/000000000139.jpg'))
    # print(image2.shape)
    # print(image2[0][0])


    # print(img3[0][0])
    # print(type(img3[0][0][0]))
    # cv2.imshow('', img3)
    # cv2.waitKey(0)

    # train = MscocoMulti(cfg, train=True)
    # img, target, valid, mata = train[0]
    # print(valid)
    # print(mata['GT_bbox'])
    # mean = cfg.pixel_means
    # img = img * 255
    # for i, im in enumerate(img):
    #     im = im.add_(mean[i])
    # img = np.transpose(np.array(img), (1, 2, 0))
    # img = np.array(img, dtype=np.uint8)

    # for t in target:
    #     print(t.shape)
    #     im = t[2]
    #     im = im * 255
    #     print(im.shape)
    #     im = np.array(im, dtype=np.uint8)
    #     im=Image.fromarray(im)
    #     im.show()

    # cv2.imshow('', img)
    # cv2.waitKey(0)
    # img = Image.fromarray(img)
    # img.show()
