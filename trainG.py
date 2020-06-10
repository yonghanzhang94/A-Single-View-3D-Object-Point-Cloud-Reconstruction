from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from os.path import join, exists, isdir, dirname, abspath, basename
import json
from datasets import GetShapenetDataset
import torch.backends.cudnn as cudnn
from model import generator
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
from loss import batch_NN_loss, batch_EMD_loss
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
torch.set_printoptions(profile='full')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
# ALL: ['02691156','02828884','02933112','02958343','03001627','03636649','03211117','04090263','04256520','03691459','04379243','04401088','04530566']
parser.add_argument('--cats', default=['02691156'], type=str,
                    help='Category to train on : ["airplane":02691156, "bench":02828884, "cabinet":02933112, '
                         '"car":02958343, "chair":03001627, "lamp":03636649, '
                         '"monitor":03211117, "rifle":04090263, "sofa":04256520, '
                         '"speaker":03691459, "table":04379243, "telephone":04401088, '
                         '"vessel"：04530566]')
parser.add_argument('--num_points', type=int, default=1024, help='number of epochs to train for, [1024, 2048]')
parser.add_argument('--outf', type=str, default='model',  help='output folder')
parser.add_argument('--modelG', type=str, default = '', help='generator model path')
parser.add_argument('--lr', type=float, default = '0.00005', help='learning rate')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)

with open(join('data/splits/', 'train_models.json'), 'r') as f:
    train_models_dict = json.load(f)

with open(join('data/splits/', 'val_models.json'), 'r') as f:
    val_models_dict = json.load(f)

data_dir_imgs = 'data/shapenet/ShapeNetRendering/'
data_dir_pcl = 'data/shapenet/ShapeNet_pointclouds/'

dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, train_models_dict, opt.cats, opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, val_models_dict, opt.cats, opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

cudnn.benchmark = True
print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

gen = generator(num_points=opt.num_points)

if not opt.modelG == '':
    with open(opt.modelG, "rb") as f:
        gen.load_state_dict(torch.load(f))

gen.cuda()

# optimizerG = optim.RMSprop(gen.parameters(), lr = opt.lr)
optimizerG = optim.Adam(gen.parameters(), lr = opt.lr)

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch+1):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):

        if i >= len(dataloader):
            break

        data = data_iter.next()
        i += 1

        images, points = data

        points = Variable(points.float())
        points = points.cuda()

        images = Variable(images.float())
        images = images.cuda()

        optimizerG.zero_grad()

        fake, _, _, _ = gen(images)
        fake = fake.transpose(2, 1)

        lossG1 = batch_NN_loss(points, fake)
        lossG2 = batch_EMD_loss(points, fake)

        lossG = lossG1 + lossG2

        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('[%d: %d/%d] train lossG: %f' %(epoch, i, num_batch, lossG.item()))

    if epoch % 20 == 0 and epoch != 0:
        opt.lr = opt.lr * 0.5
        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lr
        print('lr decay:', opt.lr)

    if epoch % 50 == 0:
        torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (opt.outf, epoch))
