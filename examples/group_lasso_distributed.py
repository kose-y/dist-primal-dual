import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import dist_pd.partitioners
import dist_pd.utils
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time
from argparser import parser_distributed

default_ngrps=1000
args = parser_distributed("Overlapping group lasso, distributed mode.", 5, "../data/ogrp_1000_130_10_5000", 414.8410**2, 1100, 100, False)

ngpu = args.ngpus
nslices = args.nslices

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(ngpu)]) # select devices

if not args.use_cpu:
    devices=sum([nslices*['/gpu:%d' % (i,)] for i in range(ngpu)], [])
else:
    devices=(['/cpu:0'])
print("device list: {}".format(devices))

iters = args.iters
interval = args.interval

import tensorflow as tf
dat = scipy.io.loadmat('{}.mat'.format(args.data_prefix))
Xdat = h5py.File('{}_X.mat'.format(args.data_prefix))
print ("loading data...")
At = np.asarray(Xdat['X'])
print ("done loading data")
n = At.shape[0]
ng = int(dat['ng'])
b = dat['Y']
g = np.arange(int(dat['ng'])).repeat(int(dat['g_size'])).reshape((-1,1))
D = dat['C'].tocoo().astype('float32')

lam = ng/100
loss = dist_pd.losses.QuadLoss


dnorm=1.4142
Lf = args.L

tau = 2*0.9/Lf
sigma = 0.9/(tau*dnorm**2)
lam = 1.0

gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )

penalty = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    prob = dist_pd.primal_dual.BaseLV(loss, penalty, At, D_dist, b, 
        tau=tau, sigma=sigma, dtype=tf.float32, devices=devices, aggregate=not args.nonergodic)
    rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True)
