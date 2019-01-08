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
parser = parser_distributed("Overlapping group lasso, distributed mode.", 1000, 5, [1000, 5000])
args = parser.parse_args()

ngpu = args.ngpus
ngrps = args.ngrps
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
dat = scipy.io.loadmat('../data/ogrp_%d_130_10_5000.mat'% (ngrps,))
Xdat = h5py.File('../data/ogrp_%d_130_10_5000_X.mat' % (ngrps,))
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
xnormdict = {}
xnormdict[1000] = 414.8410
xnormdict[5000] = 841.8572
xnormdict[8000] = 1046.9 
xnormdict[10000] = 1162.5 # for 10000 groups

xnorm = xnormdict[ngrps]

dnorm=1.4142; Lf = xnorm**2

tau = 2*0.9/Lf
sigma = 0.9/(tau*dnorm**2)
lam = 1.0

gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )

penalty = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    prob = dist_pd.primal_dual.BaseLV(loss, penalty, At, D_dist, b, 
        tau=tau, sigma=sigma, dtype=tf.float32, devices=devices)
    rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True)
