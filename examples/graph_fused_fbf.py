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

args = parser_distributed("Graph-guided fused lasso, FBF splitting.", 1, "../data/Zhu_1000_10_5000_20_0.7_100", 255.6824**2, default_output_prefix="ggfl_fbf_")
ngpu = args.ngpus
nslices = args.nslices #slices per gpu. the more the slice, the less memory usage.

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
Xdat = scipy.io.loadmat('{}_X.mat'.format(args.data_prefix ))
print ("loading data...")
At = np.asarray(Xdat['X'])
print ("done loading data")
n = At.shape[0]
b = dat['Y']
D = dat['D'].tocoo().astype('float32')

loss = dist_pd.losses.QuadLoss

dnorm=4.5074; 
Lf = args.L
tau = 2*0.9/Lf

kappas =[0]
expr_names = ['{}001_001'.format(args.output_prefix)]


lam = 1.0
penalty = dist_pd.penalties.L1Penalty(lam)

beta = Lf + dnorm
S = 0.01
alpha1 = 0.01
alpha2 = 0.01
tau = 1/beta*np.sqrt((1-12*alpha2**2-9*(alpha1+alpha2)-4*S)/(12*alpha2**2+8*(alpha1+alpha2)+4*S+2))
params = [(alpha1, alpha2, tau)]

D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices )
with tf.Session() as sess:
    for param, kappa, expr_name in zip(params, kappas, expr_names):
        alpha1, alpha2, tau = param
        prob = dist_pd.primal_dual.FBFInertial(loss, penalty, At, D_dist, b, aggregate=not args.nonergodic, 
            tau=tau, alpha1=alpha1, alpha2=alpha2, dtype=tf.float32, devices=devices)
        rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True, outfile=expr_name)

