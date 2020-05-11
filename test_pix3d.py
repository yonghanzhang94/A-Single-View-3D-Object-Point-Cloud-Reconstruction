from __future__ import print_function
import argparse
import os
import random
import torch
from os.path import join, exists, isdir, dirname, abspath, basename
import json
from datasets import GetPix3dDataset
import torch.backends.cudnn as cudnn
from model import generator
from torch.autograd import Variable
from metrics_utils import get_rec_metrics
import tensorflow as tf
from icp import icp
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--cats', default='chair', type=str,
                    help='Category to train on : ["chair","sofa","table"]')
parser.add_argument('--num_points', type=int, default=1024, help='number of pointcloud')
parser.add_argument('--model', type=str, default='./model/chair-1024/modelG_50.pth',  help='generator model path')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

gen = generator(num_points=opt.num_points)
with open(opt.model, "rb") as f:
    gen.load_state_dict(torch.load(f))
gen.cuda()
gen.eval()

with open(join('data/splits/', 'pix3d.json'), 'r') as f:
    pix3d_models_dict = json.load(f)

data_dir = './data/pix3d/'

test_dataset = GetPix3dDataset(data_dir, pix3d_models_dict, opt.cats, opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(test_dataset))

cds = []
emds = []

with torch.no_grad():
    data_iter = iter(testdataloader)
    i = 0
    try:
        while i < len(testdataloader):
            data = data_iter.next()
            i += 1

            if i >= len(testdataloader):
                break

            images, points = data

            images = Variable(images.float())
            points = Variable(points.float())

            images = images.cuda()
            points = points.cuda()

            fake, _, _, _ = gen(images)
            fake = fake.transpose(2, 1)  # b x n x c

            fake = fake.cpu().detach().numpy()
            points = points.cpu().detach().numpy()

            _pr_scaled_icp = []

            for index in range(fake.shape[0]):
                T, _, _ = icp(points[index], fake[index], tolerance=1e-10, max_iterations=1024)
                _pr_scaled_icp.append(np.matmul(fake[index], T[:3, :3]) - T[:3, 3])

            fake = np.array(_pr_scaled_icp).astype('float32')
            fake = tf.convert_to_tensor(fake)
            points = tf.convert_to_tensor(points)

            _, _, cd, emd = get_rec_metrics(points, fake, batch_size=fake.shape[0], num_points=1024)

            cd = tf.reduce_mean(cd)
            cds.append(cd)
            cdPrint = tf.Print(cd, [cd])

            emd = tf.reduce_mean(emd)
            emds.append(emd)
            emdPrint = tf.Print(emd, [emd])

            with tf.Session() as sess:
                sess.run(cdPrint)
                sess.run(emdPrint)
    except StopIteration:
        print("exception")


total_chamfer_loss = tf.reduce_mean(cds)
total_chamfer_print = tf.Print(total_chamfer_loss, [total_chamfer_loss], summarize=64)

total_emd_loss = tf.reduce_mean(emds)
total_emd_print = tf.Print(total_emd_loss, [total_emd_loss], summarize=64)

with tf.Session() as sess:
    print('chamfer distance:')
    sess.run(total_chamfer_print)
    print('earth movers distance:')
    sess.run(total_emd_print)