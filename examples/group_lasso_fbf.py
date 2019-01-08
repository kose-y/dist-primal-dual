import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import dist_pd.utils
import dist_pd.partitioners
import importlib
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time

from argparser import parser_singledevice

default_ngrps=1000
parser = parser_singledevice("Overlapping group lasso, FBF splitting.")
args = parser.parse_args()

ngpu = 1
nslices = 1

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(ngpu)]) # select devices

if not args.use_cpu:
    devices=sum([nslices*['/gpu:%d' % (i,)] for i in range(ngpu)], [])
else:
    devices=(['/cpu:0'])
print("device list: {}".format(devices))

iters = args.iters
interval = args.interval

import tensorflow as tf
dat = scipy.io.loadmat('../data/ogrp_100_100_10_5000.mat')
Xdat = h5py.File('../data/ogrp_100_100_10_5000_X.mat')
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

dnorm=1.4142; Lf = xnorm**2
kappas = [0]
expr_names = ['ogl_fbf_001_001']
params = []
tau = 2*0.9/Lf

beta = Lf + dnorm
S = 0.01
alpha1 = 0.01
alpha2 = 0.01
tau = 1/beta*np.sqrt((1-12*alpha2**2-9*(alpha1+alpha2)-4*S)/(12*alpha2**2+8*(alpha1+alpha2)+4*S+2))
params = [(alpha1, alpha2, tau)]

gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )
penalty = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    print("maxynorm: ", sess.run(penalty.maxynorm)) 
    for param, kappa, expr_name in zip(params, kappas, expr_names):
        alpha1, alpha2, tau = param
        prob = dist_pd.primal_dual.FBFInertial(loss, penalty, At, D_dist, b, aggregate=False, 
            tau=tau, alpha1=alpha1, alpha2=alpha2, dtype=tf.float32, devices=devices)
        rslt = prob.solve(check_interval=interval, max_iters=iters,outfile=expr_name,verbose=True)
