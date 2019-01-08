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
from argparser import parser_singledevice

parser = parser_singledevice("Graph-guided fused lasso, FBF splitting.")
args = parser.parse_args()
ngpu = 1
nslices = 1 #slices per gpu. the more the slice, the less memory usage.

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(ngpu)]) # select devices

if not args.use_cpu:
    devices=sum([nslices*['/gpu:%d' % (i,)] for i in range(ngpu)], [])
else:
    devices=(['/cpu:0'])
print("device list: {}".format(devices))

iters = args.iters
interval = args.interval

import tensorflow as tf
dat = scipy.io.loadmat('../data/Zhu_1000_10_5000_20_0.7_100.mat' )
Xdat = scipy.io.loadmat('../data/Zhu_1000_10_5000_20_0.7_100_X.mat' )
print ("loading data...")
At = np.asarray(Xdat['X'])
print ("done loading data")
n = At.shape[0]
b = dat['Y']
D = dat['D'].tocoo().astype('float32')

loss = dist_pd.losses.QuadLoss

xnorm = 255.6824
dnorm=4.5074; Lf = xnorm**2
tau = 2*0.9/Lf

kappas =[0]
expr_names = ['ggfl_fbf_001_001']


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
        prob = dist_pd.primal_dual.FBFInertial(loss, penalty, At, D_dist, b, aggregate=False, 
            tau=tau, alpha1=alpha1, alpha2=alpha2, dtype=tf.float32, devices=devices)
        rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True, outfile=expr_name)

