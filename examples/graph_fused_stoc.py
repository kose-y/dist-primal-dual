import os

import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
from dist_pd.optimal_paramset import BddStocParamSet, UnbddStocParamSet

import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time

from argparser import parser_distributed

args = parser_distributed("Graph-guided fused lasso, stochastic iteration.", 1, "../data/Zhu_1000_10_5000_20_0.7_100", 255.6824**2, default_output_prefix="ggfl_stoc_", stoc=True, default_s=10000000)
ngpu = args.ngpus
nslices = args.nslices

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(ngpu)]) # select devices

if not args.use_cpu:
    devices=sum([nslices*['/gpu:%d' % (i,)] for i in range(ngpu)], [])
else:
    devices=(['/cpu:0'])
print("device_list: {}".format(devices))

iters = args.iters
interval = args.interval

import tensorflow as tf
tf.set_random_seed(1234)
dat = scipy.io.loadmat('{}.mat'.format(args.data_prefix) )
Xdat = scipy.io.loadmat('{}_X.mat'.format(args.data_prefix) )
print ("loading data...")
At = np.asarray(Xdat['X'])
print ("done loading data")
n = At.shape[0]
b = dat['Y']
D = dat['D'].tocoo().astype('float32')

loss = dist_pd.losses.QuadLoss
#xnorm = 255.6824
dnorm=4.5074
Lf = args.L
tau = 2*0.9/Lf
sigma = 0.9/(tau*dnorm**2)
lam = 1.0
penalty = dist_pd.penalties.L1Penalty(lam)
Omega_X = np.sqrt(2*100**2)
Omega_Y = np.sqrt(2*46800)

s_base = args.s
dtilde = 100
s = s_base/dtilde

params_list = []
ab_list = [(-1, 0), (0, 0), (-0.5, 0.5), (-1, 1), (-1, 0), (0, 0), (-0.5, 0.5), (-1, 1) ]
expr_names = ['bdd_chen','bdd_lv', 'bdd_half', 'bdd_cv',  'unbdd_chen', 'unbdd_lv', 'unbdd_half', 'unbdd_cv']
expr_names = [args.output_prefix+x for x in expr_names]

params_list.append(BddStocParamSet(iters, 1, 0, 0, 1, Lf, dnorm, Omega_X, Omega_Y, s_base, s_base)) # Chen
params_list.append(BddStocParamSet(iters, 0, 0, 1, 1, Lf, dnorm, Omega_X, Omega_Y, s_base, s_base)) # LV
params_list.append(BddStocParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm, Omega_X, Omega_Y, s_base, s_base)) # mid
params_list.append(BddStocParamSet(iters, 1, 1, 0, 2, Lf, dnorm, Omega_X, Omega_Y, s_base, s_base))# CV

params_list.append(UnbddStocParamSet(iters, 1, 0, 0, 1, Lf, dnorm, s)) # Chen
params_list.append(UnbddStocParamSet(iters, 0, 0, 1, 1, Lf, dnorm, s)) # LV
params_list.append(UnbddStocParamSet(iters, 0.5, 0.5, 0.5, 1.5, Lf, dnorm, s)) # mid
params_list.append(UnbddStocParamSet(iters, 1, 1, 0, 2, Lf, dnorm, s))# CV

D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices )
with tf.Session() as sess:
    for params, ab, expr_name in zip(params_list, ab_list, expr_names):
        print(expr_name) 
        prob = dist_pd.primal_dual.OptimalAB(loss, penalty, At, D_dist, b, 
            tau=params.tau, sigma=params.sigma, rho=params.rho, theta=params.theta, 
            coef_a = ab[0], coef_b=ab[1], 
            dtype=tf.float32, devices=devices, aggregate=not args.nonergodic)
        rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True, outfile=expr_name, return_var=False)

