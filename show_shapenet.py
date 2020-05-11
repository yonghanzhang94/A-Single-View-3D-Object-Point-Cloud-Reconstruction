from __future__ import print_function
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import generator
from torch.autograd import Variable
import cv2
import matplotlib.pylab as plt
from icp import icp
import tensorflow as tf
from metrics_utils import get_rec_metrics
from mpl_toolkits.mplot3d import Axes3D

base_path = './data/shapenet/'
file_path = '02691156/1a04e3eab45ca15dd86060f189eb133'
img_path = base_path + 'ShapeNetRendering/' + file_path + '/rendering/00.png'
point_path = base_path + 'ShapeNet_pointclouds/' + file_path + '/pointcloud_2048.npy'
pickle_path = './model/airplane-2048/modelG_50.pth'

cudnn.benchmark = True

image = cv2.imread(img_path)[4:-5, 4:-5, :3]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.transpose(image, (2, 0, 1))
image = torch.Tensor(image)
image = image.unsqueeze(0)
# print(image)
points_gt = (np.load(point_path)).astype('float32')    # groundtruth

gen = generator(num_points=1024)
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
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)
for i in range(len(points)):
    ax.scatter(points[i, 1], points[i, 2], points[i, 0], c='#00008B', depthshade=True)
# for i in range(len(points_gt)):
#     ax.scatter(points_gt[i, 1], points_gt[i, 2], points_gt[i, 0], c='#00008B', depthshade=True)    # show groundtruth
ax.axis('off')
ax.view_init(azim=90, elev=-160)
plt.show()

# # calc CD and EMD
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
