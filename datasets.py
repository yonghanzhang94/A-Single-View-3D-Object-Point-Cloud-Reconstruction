import torch.utils.data as data
from os.path import join, exists, isdir, dirname, abspath, basename
import json
import torch
import numpy as np
import cv2
import os

NUM_VIEWS = 24
images = []
HEIGHT = 128
WIDTH = 128
PAD = 35

class GetShapenetDataset(data.Dataset):
    def __init__(self, data_dir_imgs, data_dir_pcl, models, cats, numpoints=1024, variety=False):
        self.data_dir_imgs = data_dir_imgs
        self.data_dir_pcl = data_dir_pcl
        self.models = models
        self.modelnames = []
        self.size = 0
        self.numpoints = numpoints
        self.variety = variety

        for cat in cats:
            for filename in self.models[cat]:
                for i in range(NUM_VIEWS):
                    self.size = self.size + 1
                    self.modelnames.append(filename)

    def __getitem__(self, index):
        img_path = self.data_dir_imgs + self.modelnames[index] + '/rendering/' + (str(int(index % NUM_VIEWS)).zfill(2) + '.png')
        # print(imagePath)
        image = cv2.imread(img_path)[4:-5, 4:-5, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))

        pcl_path = self.data_dir_pcl + self.modelnames[index] + '/pointcloud_' + str(self.numpoints) + '.npy'
        pcl_gt = np.load(pcl_path)

        if self.variety == True:
            metadata_path = self.data_dir_imgs + self.modelnames[index] + '/rendering/rendering_metadata.txt'
            metadata = np.loadtxt(metadata_path)
            x = metadata[(int(index % NUM_VIEWS))][0]
            xangle = np.pi / 180. * x
            y = metadata[(int(index % NUM_VIEWS))][1]
            yangle = np.pi / 180. * y
            return image, pcl_gt, xangle, yangle

        return image, pcl_gt

    def __len__(self):
        return self.size


class GetPix3dDataset(data.Dataset):
    def __init__(self, data_dir, models, cats, numpoints=1024, save=False):
        self.save = save
        self.data_dir = data_dir
        self.models = models
        self.size = 0
        self.cats = cats
        self.numpoints = numpoints
        self.imgpaths = []
        self.maskpaths = []
        self.modelpaths = []
        self.bbox = []
        pcl = 'pcl_' + str(self.numpoints)

        for model in self.models:
            if model['category'] == self.cats:
                # model/[category]/[modelname] /model.obj
                modelpath = model['model'].replace("model", pcl)  # pcl_1024/[category]/[modelname]/pcl_1024.obj
                modelpath = modelpath.replace("pcl_1024", "model", 1)  # model/[category]/[modelname]/pcl_1024.obj
                modelpath = modelpath.replace("obj", 'npy')  # model/[category]/[modelname]/pcl_1024.npy
                pcl_path = self.data_dir + 'pointclouds/' + modelpath
                if os.path.exists(pcl_path):
                    self.imgpaths.append(model['img'])
                    self.maskpaths.append(model['mask'])
                    self.modelpaths.append(model['model'])
                    self.bbox.append(model['bbox'])
                    self.size = self.size + 1

    def __getitem__(self, index):
        img_path = self.data_dir + self.imgpaths[index]
        mask_path = self.data_dir + self.maskpaths[index]
        pcl = 'pcl_' + str(self.numpoints)
        modelpath = self.modelpaths[index].replace("model", pcl)
        modelpath = modelpath.replace("pcl_1024", "model", 1)
        modelpath = modelpath.replace("obj", 'npy')
        pcl_path = self.data_dir + 'pointclouds/' + modelpath
        img_name = img_path[-8:-4]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path)
        if not image.shape[0] == mask_image.shape[0] or not image.shape[1] == mask_image.shape[1]:
            mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
        image = image * mask_image
        image = image[self.bbox[index][1]:self.bbox[index][3], self.bbox[index][0]:self.bbox[index][2], :]
        current_size = image.shape[:2]
        ratio = float(HEIGHT - PAD) / max(current_size)
        new_size = tuple([int(x * ratio) for x in current_size])
        image = cv2.resize(image, (new_size[1], new_size[0]))  # new_size should be in (width, height) format
        delta_w = WIDTH - new_size[1]
        delta_h = HEIGHT - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        image = np.transpose(image, (2, 0, 1))

        xangle = np.pi / 180. * -90
        yangle = np.pi / 180. * -90
        pcl_gt = rotate(rotate(np.load(pcl_path), xangle, yangle), xangle)
        if not self.save:
            return image, pcl_gt
        else:
            return image, pcl_gt, img_name

    def __len__(self):
        return self.size


def rotate(xyz, xangle=0, yangle=0, zangle=0):
    rotmat = np.eye(3)
    rotmat = rotmat.dot(np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(xangle), -np.sin(xangle)],
        [0.0, np.sin(xangle), np.cos(xangle)],
    ]))
    rotmat = rotmat.dot(np.array([
        [np.cos(yangle), 0.0, -np.sin(yangle)],
        [0.0, 1.0, 0.0],
        [np.sin(yangle), 0.0, np.cos(yangle)],
    ]))
    rotmat = rotmat.dot(np.array([
        [np.cos(zangle), -np.sin(zangle), 0.0],
        [np.sin(zangle), np.cos(zangle), 0.0],
        [0.0, 0.0, 1.0]
    ]))

    return xyz.dot(rotmat)


