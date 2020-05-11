from __future__ import division
import os
import sys
import numpy as np
import tensorflow as tf
import time
from os import listdir, makedirs
from os.path import join, exists, isdir, dirname, abspath, basename
import csv
# import ipdb

from tf_ops.cd import tf_nndistance
from tf_ops.emd.tf_auctionmatch import auction_match


def get_rec_metrics(gt_pcl, pred_pcl, batch_size=16, num_points=1024):
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl, pred_pcl)
    dists_forward = tf.reduce_mean(tf.sqrt(dists_forward), axis=1)  # (B, )
    dists_backward = tf.reduce_mean(tf.sqrt(dists_backward), axis=1)  # (B, )
    chamfer_distance = dists_backward + dists_forward

    X, _ = tf.meshgrid(tf.range(batch_size), tf.range(num_points), indexing='ij')
    ind, _ = auction_match(pred_pcl, gt_pcl)  # Ind corresponds to points in pcl_gt
    ind = tf.stack((X, ind), -1)
    emd = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.gather_nd(gt_pcl, ind) - pred_pcl) ** 2, axis=-1)),
                         axis=1)  # (BATCH_SIZE,NUM_POINTS,3) --> (BATCH_SIZE,NUM_POINTS) --> (BATCH_SIZE)

    return dists_forward, dists_backward, chamfer_distance, emd