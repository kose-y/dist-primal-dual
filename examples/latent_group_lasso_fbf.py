import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.split_22
import dist_pd.distmat
import dist_pd.utils
import dist_pd.partitioners
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time

from argparser import parser_distributed

args = parser_distributed("Latent group lasso, FBF splitting.", 1, "../data/ogrp_100_100_10_5000", 164.9960**2, default_output_prefix="lgl_fbf_")

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
dat = scipy.io.loadmat('{}.mat'.format(args.data_prefix) )
Xdat = h5py.File('{}_X.mat'.format(args.data_prefix ))
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
xnorm = 164.9960 # for 100 grps

dnorm=2; Lf = args.L
kappas=[0]
expr_names = ['{}001_001'.format(args.output_prefix)]
params = []
expr_name = None
tau = 2*0.9/Lf
 
beta = Lf + dnorm
S = 0.01
alpha1 = 0.01
alpha2 = 0.01
tau = 1/beta*np.sqrt((1-12*alpha2**2-9*(alpha1+alpha2)-4*S)/(12*alpha2**2+8*(alpha1+alpha2)+4*S+2))
params = [(alpha1, alpha2, tau)]

gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )
penalty1 = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)
penalty2 = dist_pd.penalties.Ind0()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    for param, kappa, expr_name in zip(params, kappas, expr_names):
        alpha1, alpha2, tau = param
        prob = dist_pd.split_22.FBFInertial(loss, penalty1, penalty2, At, D_dist, b, aggregate=not args.nonergodic,
            tau=tau, alpha1=alpha1, alpha2=alpha2, dtype=tf.float32, devices=devices)
        rslt = prob.solve(check_interval=interval, max_iters=iters,outfile=expr_name,verbose=True)
