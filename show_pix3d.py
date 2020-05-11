from __future__ import print_function
from os.path import join
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import generator
from torch.autograd import Variable
import cv2
import matplotlib.pylab as plt
from datasets import rotate
from icp import icp
import tensorflow as tf
from metrics_utils import get_rec_metrics
from mpl_toolkits.mplot3d import Axes3D

cudnn.benchmark = True

with open(join('data/splits/', 'pix3d.json'), 'r') as f:
    pix3d_models_dict = json.load(f)

data_dir = './data/pix3d/'
# ["chair","sofa","table"]
model = 'chair'
file_name = '0001'
img_path = 'img/' + model + '/' + file_name + '.png'
mask_path = 'mask/' + model + '/' + file_name + '.png'
pickle_path = './model/chair-2048/modelG_50.pth'
gen_numpoints = 2048 

#  dataset parameter
HEIGHT = 128
WIDTH = 128
PAD = 35
numpoints = 1024
pcl = 'pcl_' + str(numpoints)

for model in pix3d_models_dict:
    if model['img'] == img_path:
        modelpath = model['model'].replace("model", pcl)  # pcl_1024/[category]/[modelname]/pcl_1024.obj
        modelpath = modelpath.replace("pcl_1024", "model", 1)  # model/[category]/[modelname]/pcl_1024.obj
        modelpath = modelpath.replace("obj", 'npy')  # model/[category]/[modelname]/pcl_1024.npy
        bbox = model['bbox']

        img_path = data_dir + img_path
        mask_path = data_dir + mask_path
        pcl_path = data_dir + 'pointclouds/' + modelpath

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_image = cv2.imread(mask_path)
        if not image.shape[0] == mask_image.shape[0] or not image.shape[1] == mask_image.shape[1]:
            mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
        image = image * mask_image
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
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
        image = torch.Tensor(image)
        image = image.unsqueeze(0)

        xangle = np.pi / 180. * -90
        yangle = np.pi / 180. * -90
        points_gt = rotate(rotate(np.load(pcl_path), xangle, yangle), xangle)

gen = generator(num_points=gen_numpoints)
gen.cuda().eval()

with open(pickle_path, "rb") as f:
    gen.load_state_dict(torch.load(f))

image = Variable(image.float())
image = image.cuda()
points, _, _, _ = gen(image)
points = points.cpu().detach().numpy()
points = np.squeeze(points)
points = np.transpose(points, (1, 0))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(-0.35, 0.35)
ax.set_ylim(-0.35, 0.35)
ax.set_zlim(-0.35, 0.35)
# for i in range(len(points)):
#     ax.scatter(points[i, 1], points[i, 2], points[i, 0], c='#00008B', depthshade=True)
for i in range(len(points_gt)):
    ax.scatter(points_gt[i, 1], points_gt[i, 2], points_gt[i, 0], c='#00008B', depthshade=True)    # show groundtruth
ax.axis('off')
ax.view_init(azim=-45, elev=-165)
plt.show()


# _pr_scaled_icp = []
# T, _, _ = icp(points_gt, points, tolerance=1e-10, max_iterations=1024)
# _pr_scaled_icp.append(np.matmul(points, T[:3, :3]) - T[:3, 3])
#
# points = np.array(_pr_scaled_icp).astype('float32')
# points = tf.convert_to_tensor(points)
# points_gt = tf.convert_to_tensor(points_gt)
# points_gt = tf.expand_dims(points_gt, 0)
#
# _, _, cd, emd = get_rec_metrics(points_gt, points, batch_size=1, num_points=1024)
#
# cdPrint = tf.Print(cd, [cd], summarize=64)
# emdPrint = tf.Print(emd, [emd], summarize=64)
#
# with tf.Session() as sess:
#     print('chamfer distance:')
#     sess.run(cdPrint)
#     print('earth mover distance:')
#     sess.run(emdPrint)