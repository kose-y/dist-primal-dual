import os
import dist_pd.losses
import dist_pd.penalties
import dist_pd.primal_dual 
import dist_pd.distmat
import dist_pd.partitioners
import dist_pd.utils
import numpy as np 
from scipy.sparse import dia_matrix
import scipy.io
import h5py
import sys
import time

from argparser import parser_singledevice

default_ngrps=1000
parser = parser_singledevice("Overlapping group lasso, FB splitting-based iterations")
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
xnorm = 164.9960


dnorm=1.4142; Lf = xnorm**2
kappas = [-1, -0.1, -0.5, 0]
expr_names = ['ogl_base_cv', 'ogl_base_01', 'ogl_base_05', 'ogl_base_lv']
params = []
tau = 2*0.9/Lf
for kappa in kappas:
    def get_params(kappa):
        sigma = 0.9/(tau*dnorm**2) * (1-tau*Lf/2)/(1-(1-kappa**2)*tau*Lf/2)
        mu = (1-sigma*tau*dnorm**2)/(1-(1-kappa**2)*sigma*tau*dnorm**2)/tau
        rho = 0.9*(2-Lf/(2*mu))
        return sigma, rho
    sigma, rho = get_params(kappa)
    params.append((tau, sigma, rho))

gpart = dist_pd.utils.gidx_to_partition(g)
D_dist = dist_pd.distmat.DistSpMat.from_spmatrix(D, devices, partitioner_r = dist_pd.partitioners.group_partitioner(gpart) )
penalty = dist_pd.penalties.GroupLasso(lam, g, devices = devices, 
    partition=D_dist.partition_r,dtype=tf.float32)
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    print("maxynorm: ", sess.run(penalty.maxynorm)) 
    for param, kappa, expr_name in zip(params, kappas, expr_names):
        tau, sigma, rho = param
        prob = dist_pd.primal_dual.BaseKappa(loss, penalty, At, D_dist, b, aggregate=True, 
            tau=tau, sigma=sigma, rho=rho, dtype=tf.float32, devices=devices, kappa=kappa)
        rslt = prob.solve(check_interval=interval, max_iters=iters,outfile=expr_name,verbose=True)

