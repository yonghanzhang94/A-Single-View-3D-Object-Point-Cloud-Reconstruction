from __future__ import print_function
import numpy as np
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

NUM_VIEWS = 1
cudnn.benchmark = True
azim = -50
elev = -145
scale = 0.4
base_path = './data/shapenet/'
category = '02691156'  # use number
save_path = './shapenet_img/' + category + '/'
pickle_path = './model/airplane-2048/modelG_50.pth'

dict_file_path={"airplane":'02691156', "bench":'02828884', "cabinet":'02933112',
                        "car":'02958343', "chair":'03001627', "lamp":'03636649',
                         "monitor":'03211117', "rifle":'04090263', "sofa":'04256520',
                         "speaker":'03691459', "table":'04379243', "telephone":'04401088',
                         "vessel":'04530566'}

if not os.path.exists(save_path):
    os.makedirs(save_path)

file_paths = os.listdir(base_path + 'ShapeNetRendering/' + category + '/')

gen = generator(num_points=2048)
gen.cuda().eval()
with open(pickle_path, "rb") as f:
    gen.load_state_dict(torch.load(f))


for file_path in file_paths:
    point_path = base_path + 'ShapeNet_pointclouds/' + category + '/' + file_path + '/pointcloud_2048.npy'
    points_gt = (np.load(point_path)).astype('float32')  # groundtruth

    # save groundtruth img
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    for i in range(len(points_gt)):
        ax.scatter(points_gt[i, 1], points_gt[i, 2], points_gt[i, 0], c='#00008B', depthshade=True)
    ax.axis('off')
    ax.view_init(azim=azim, elev=elev)
    plt.savefig(save_path + file_path + '_gt.png')
    # plt.show()
    plt.close()

    for i in range(NUM_VIEWS):
        img_path = base_path + 'ShapeNetRendering/' + category + '/' + file_path + '/rendering/' + (str(int(i % NUM_VIEWS)).zfill(2) + '.png')
        image = cv2.imread(img_path)[4:-5, 4:-5, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_path + file_path + '_' + (str(int(i % NUM_VIEWS)).zfill(2) + '.png'), image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.Tensor(image)
        image = image.unsqueeze(0)
        # print(image)

        image = Variable(image.float())
        image = image.cuda()
        points, _, _, _ = gen(image)
        points = points.cpu().detach().numpy()
        points = np.squeeze(points)
        points = np.transpose(points, (1, 0))

        # save predict img
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
        for i in range(len(points)):
            ax.scatter(points[i, 1], points[i, 2], points[i, 0], c='#00008B', depthshade=True)
        ax.axis('off')
        ax.view_init(azim=azim, elev=elev)
        plt.savefig(save_path + file_path + '_' + (str(int(i % NUM_VIEWS)).zfill(2) + '_pr.png'))
        # plt.show()
        plt.close()

    print("saving model img...")