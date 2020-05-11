from __future__ import print_function
import argparse
from os.path import join, exists, isdir, dirname, abspath, basename
import json
import numpy as np
from datasets import GetPix3dDataset
import torch
import torch.backends.cudnn as cudnn
from model import generator
from torch.autograd import Variable
import cv2
import matplotlib.pylab as plt
import os
from icp import icp
import tensorflow as tf
from metrics_utils import get_rec_metrics
from mpl_toolkits.mplot3d import Axes3D

cudnn.benchmark = True
azim = -45
elev = -165
scale = 0.45

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')  #save img batchsize = 1
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--cats', default='chair', type=str,
                    help='Category to train on : ["chair","sofa","table"]')
parser.add_argument('--num_points', type=int, default=2048, help='number of pointcloud')
parser.add_argument('--model', type=str, default='./model/chair-2048/modelG_50.pth',  help='generator model path')
opt = parser.parse_args()

save_path = './pix3d_img/' + opt.cats + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

gen = generator(num_points=opt.num_points)
gen.cuda().eval()
with open(opt.model, "rb") as f:
    gen.load_state_dict(torch.load(f))

with open(join('data/splits/', 'pix3d.json'), 'r') as f:
    pix3d_models_dict = json.load(f)
data_dir = './data/pix3d/'
test_dataset = GetPix3dDataset(data_dir, pix3d_models_dict, opt.cats, 1024, save=True)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))
print(len(test_dataset))

with torch.no_grad():
    data_iter = iter(testdataloader)
    index = 0
    while index < len(testdataloader):
        data = data_iter.next()
        index += 1

        if index >= len(testdataloader):
            break

        images, points, img_name = data

        images = Variable(images.float())
        points = Variable(points.float())

        images = images.cuda()
        points = points.cuda()

        fake, _, _, _ = gen(images)
        fake = fake.transpose(2, 1)  # b x n x c

        fake = np.squeeze(fake.cpu().detach().numpy())  # n x c
        points = np.squeeze(points.cpu().detach().numpy())  # n x c

        # save groundtruth img
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
        for i in range(len(points)):
            ax.scatter(points[i, 1], points[i, 2], points[i, 0], c='#00008B', depthshade=True)
        ax.axis('off')
        ax.view_init(azim=azim, elev=elev)
        plt.savefig(save_path + img_name[0] + '_gt.png')
        # plt.show()
        plt.close()

        # save predict img
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
        for i in range(len(fake)):
            ax.scatter(fake[i, 1], fake[i, 2], fake[i, 0], c='#00008B', depthshade=True)
        ax.axis('off')
        ax.view_init(azim=azim, elev=elev)
        plt.savefig(save_path + img_name[0] + '_pr.png')
        # plt.show()
        plt.close()

        if index % 5 == 0:
            print("saving " + str(index) + " imgs")