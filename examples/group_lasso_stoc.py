import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import dist_pd.utils
import dist_pd.partitioners
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time
from dist_pd.optimal_paramset import BddStocParamSet, UnbddStocParamSet

from argparser import parser_singledevice

parser = parser_singledevice("Overlapping group lasso, stochastic iteration.")
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

loss = dist_pd.losses.QuadLoss
xnorm = 164.9960 # for 100 grps
dnorm=1.4142; Lf = xnorm**2
tau = 2*0.9/Lf
sigma = 0.9/(tau*dnorm**2)
lam = ng/100.0
#penalty=dist_pd.penalties.GroupLasso(lam, g, dtype=tf.float32)
ab_list = [(-1, 0), (0, 0), (-0.5, 0.5), (-1, 1)]
expr_names = ['ogl_bdd_chen_stoc', 'ogl_bdd_lv_stoc', 'ogl_bdd_half_stoc', 'ogl_bdd_cv_stoc']
expr_names += ['ogl_unbdd_chen_stoc', 'ogl_unbdd_lv_stoc', 'ogl_unbdd_half_stoc', 'ogl_unbdd_cv_stoc']
params_list = []
p = 0.95
Omega_X = 12
Omega_Y = 15
dinfnorm = 2
s_b = 300000
s = 300000
print("s: ", s)
dtilde = 10
s/= dtilde
expr_names = [x+'_'+str(dtilde)  for x in expr_names]

params_list.append(BddStocParamSet(iters, 1, 0, 0, 1, Lf, dnorm, Omega_X, Omega_Y, s_b, s_b)) # Chen
params_list.append(BddStocParamSet(iters, 0, 0, 1, 1, Lf, dnorm, Omega_X, Omega_Y, s_b, s_b)) 
params_list.append(BddStocParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm, Omega_X, Omega_Y, s_b, s_b)) 
params_list.append(BddStocParamSet(iters, 1, 1, 0, 2, Lf, dnorm, Omega_X, Omega_Y, s_b, s_b))

params_list.append(UnbddStocParamSet(iters, 1, 0, 0, 1, Lf, dnorm, s)) # Chen
params_list.append(UnbddStocParamSet(iters, 0, 0, 1, 1, Lf, dnorm, s)) 
params_list.append(UnbddStocParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm, s)) 
params_list.append(UnbddStocParamSet(iters, 1, 1, 0, 2, Lf, dnorm, s))

print("lam", lam)
gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )
penalty = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32) 
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    for params, ab, expr_name in zip(params_list, ab_list, expr_names):     

        prob = dist_pd.primal_dual.OptimalABStochastic(loss, penalty, 0.2, 1.0, At, D_dist, b, 
            tau=params.tau, sigma=params.sigma, rho=params.rho, theta=params.theta, 
            coef_a=ab[0], coef_b=ab[1],
            dtype=tf.float32, devices=devices, aggregate=True, sess=sess)
        rslt = prob.solve(check_interval=interval, max_iters=iters, verbose=True, outfile=expr_name)
    
    
    

