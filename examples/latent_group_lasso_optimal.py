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
from dist_pd.optimal_paramset import BddParamSet, UnbddParamSet
from argparser import parser_singledevice

default_ngrps=1000
parser = parser_singledevice("Latent group lasso, optimal-rate iteration.")
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

Omega_X = 12
Omega_Y = 15


gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r=dist_pd.partitioners.group_partitioner(gpart) )

penalty1 = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)
penalty2 = dist_pd.penalties.Ind0()


dnorm=2; Lf = xnorm**2
ab_list = [(-1, 0), (0, 0), (-0.5, 0.5), (-1, 1), (-1, 0), (0, 0), (-0.5, 0.5), (-1, 1) ]
expr_names = ['lgl_bdd_chen', 'lgl_bdd_lv', 'lgl_bdd_half', 'lgl_bdd_cv', 'lgl_unbdd_chen', 'lgl_unbdd_lv', 'lgl_unbdd_half', 'lgl_unbdd_cv']

params_list = []
params_list.append(BddParamSet(iters, 1, 0, 0, 1, Lf, dnorm, Omega_X, Omega_Y)) # Chen
params_list.append(BddParamSet(iters, 0, 0, 1, 1, Lf, dnorm, Omega_X, Omega_Y)) # LV
params_list.append(BddParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm, Omega_X, Omega_Y)) # mid
params_list.append(BddParamSet(iters, 1, 1, 0, 2, Lf, dnorm, Omega_X, Omega_Y))# CV
params_list.append(UnbddParamSet(iters, 1, 0, 0, 1, Lf, dnorm, iota=1)) # Chen
params_list.append(UnbddParamSet(iters, 0, 0, 1, 1, Lf, dnorm)) # LV
params_list.append(UnbddParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm)) # mid
params_list.append(UnbddParamSet(iters, 1, 1, 0, 2, Lf, dnorm))# CV
    

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    for params, ab, expr_name in zip(params_list, ab_list, expr_names):
        prob = dist_pd.split_22.OptimalAB(loss, penalty1, penalty2, At, D_dist, b, aggregate=True, 
            tau=params.tau, sigma=params.sigma, rho=params.rho, theta=params.theta, 
            coef_a = ab[0], coef_b = ab[1], 
            dtype=tf.float32, devices=devices, sess=sess)
        rslt = prob.solve(check_interval=interval, max_iters=iters,outfile=expr_name,verbose=True, return_var=False)
