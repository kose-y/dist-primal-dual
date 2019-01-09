import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import dist_pd.utils
import dist_pd.partitioners
from dist_pd.optimal_paramset import BddParamSet, UnbddParamSet
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time

from argparser import parser_distributed

args = parser_distributed("Overlapping group lasso, optimal-rate iteration.", 1, "../data/ogrp_100_100_10_5000", 164.9960**2, default_output_prefix="ogl_opt_")

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

#single device version
loss = dist_pd.losses.QuadLoss
#xnorm = 164.9960 # for 100 grps

dnorm=1.4142
Lf = args.L


tau = 2*0.9/Lf
sigma = 0.9/(tau*dnorm**2)
lam = ng/100.0 

Omega_X = 12
Omega_Y = 15


ab_list = [(-1, 0), (0, 0), (-0.5, 0.5), (-1, 1), (-1, 0), (0, 0), (-0.5, 0.5), (-1, 1) ]
expr_names = ['bdd_chen', 'bdd_lv', 'bdd_half', 'bdd_cv', 'unbdd_chen', 'unbdd_lv', 'unbdd_half', 'unbdd_cv']
expr_names = [args.output_prefix+x for x in expr_names]

params_list = []
# Chen
params_list.append(BddParamSet(iters, 1, 0, 0, 1, Lf, dnorm, Omega_X, Omega_Y)) # Chen
params_list.append(BddParamSet(iters, 0, 0, 1, 1, Lf, dnorm, Omega_X, Omega_Y)) # LV
params_list.append(BddParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm, Omega_X, Omega_Y)) # mid
params_list.append(BddParamSet(iters, 1, 1, 0, 2, Lf, dnorm, Omega_X, Omega_Y))# CV
params_list.append(UnbddParamSet(iters, 1, 0, 0, 1, Lf, dnorm, iota=1)) # Chen
params_list.append(UnbddParamSet(iters, 0, 0, 1, 1, Lf, dnorm)) # LV
params_list.append(UnbddParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm)) # mid
params_list.append(UnbddParamSet(iters, 1, 1, 0, 2, Lf, dnorm))# CV

print("lam", lam)
gpart = dist_pd.utils.gidx_to_partition(g)


D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )
penalty = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    for params, ab, expr_name in zip(params_list, ab_list, expr_names):
        prob = dist_pd.primal_dual.OptimalAB(loss, penalty, At, D_dist, b, 
                tau=params.tau, sigma=params.sigma, rho=params.rho, theta=params.theta, 
                coef_a=ab[0], coef_b=ab[1],
                dtype=tf.float32, devices=devices, aggregate=not args.nonergodic, sess=sess)
        rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True, outfile=expr_name, return_var=False)
    
    
    

