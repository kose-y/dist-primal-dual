import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import importlib
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time

from argparser import parser_distributed

args = parser_distributed("Graph-guided fused lasso, FB splitting-based iterations.", 1, "../data/Zhu_1000_10_5000_20_0.7_100", 255.6824**2, default_output_prefix="ggfl_fb_")
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
dat = scipy.io.loadmat('{}.mat'.format(args.data_prefix) )
Xdat = scipy.io.loadmat('{}_X.mat'.format(args.data_prefix) )
print ("loading data...")
At = np.asarray(Xdat['X'])
print ("done loading data")
n = At.shape[0]
b = dat['Y']
D = dat['D'].tocoo().astype('float32')

#single device version
loss = dist_pd.losses.QuadLoss
#xnorm = 255.6824
dnorm=4.5074
Lf = args.L
tau = 2*0.9/Lf

kappas =[-1, -0.1, -0.5, 0]
expr_names = ['cv', '01', '05', 'lv']
expr_names = [args.output_prefix+x for x in expr_names]
sigma = 0.9/(tau*dnorm**2)
lam = 1.0
penalty=dist_pd.penalties.L1Penalty(lam)
params = []
for kappa in kappas:
    def get_params(kappa):
        sigma = 0.9/(tau*dnorm**2) * (1-tau*Lf/2)/(1-(1-kappa**2)*tau*Lf/2)
        mu = (1-sigma*tau*dnorm**2)/(1-(1-kappa**2)*sigma*tau*dnorm**2)/tau
        rho = 0.9*(2-Lf/(2*mu))
        return sigma, rho
    sigma, rho = get_params(kappa)
    params.append((tau, sigma, rho))

D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices )
with tf.Session() as sess:
    for param, kappa, expr_name in zip(params, kappas, expr_names):
        tau, sigma, rho = param
        prob = dist_pd.primal_dual.BaseKappa(loss, penalty, At, D_dist, b, aggregate=not args.nonergodic, 
            tau=tau, sigma=sigma, dtype=tf.float32, devices=devices, kappa=kappa)
        rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True, outfile=expr_name)

