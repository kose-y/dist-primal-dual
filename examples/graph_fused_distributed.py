import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time
from argparser import parser_distributed

args = parser_distributed("Graph-guided fused lasso, distributed mode.", 5, "../data/Zhu_10000_12_5000_20_0.7_10000", 499.6337**2, 1100, 100, False)


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
dat = scipy.io.loadmat(args.data_prefix+'.mat')
Xdat = h5py.File(args.data_prefix+'_X.mat')
print ("loading data...")
At = np.asarray(Xdat['X'])
print ("done loading data")
n = At.shape[0]
b = dat['Y']
D = dat['D'].tocoo().astype('float32')

#single device version
loss = dist_pd.losses.QuadLoss
#xnormdict = {}
#xnormdict = 499.6337
#xnormdict[50000] = 919.5103 # for 50000 grps
#xnormdict[80000] = 1124.4 # for 80000 grps
#xnormdict[100000] = 1238.6 # for 100000 grps
#xnorm = xnormdict[ngrps]
dnorm=4.9045#; Lf = xnorm**2
Lf = args.L
tau = 2*0.9/Lf
sigma = 0.9/(tau*dnorm**2)
lam = 1.0
penalty = dist_pd.penalties.L1Penalty(lam)


D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices )
with tf.Session() as sess:
    prob = dist_pd.primal_dual.BaseLV(loss, penalty, At, D_dist, b, 
            tau=tau, sigma=sigma, dtype=tf.float32, devices=devices, aggregate=not args.nonergodic)
    rslt = prob.solve(check_interval=interval, max_iters=iters,verbose=True)

