from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import sparse_ops
from scipy.sparse import coo_matrix, spmatrix
from dist_pd.utils import coo_to_sparsetensor
import dist_pd.distops as distops
import dist_pd.distmat as distmat
import dist_pd.penalties as penalties


"""
split_22.py is intended for latent group lasso implementation.
"""


one = tf.constant(1.0)

class Split22:
    def __init__(self, loss, penalty1, penalty2, At, D, b, tau=None, sigma=None, sess=None, dtype=dtypes.float32, devices='', aggregate=False, init_var=None):
        if not tau:
            raise ValueError("Must set tau")
        if not sigma:
            raise ValueError("Must set sigma")
        if sess is None:
            sess = session.Session()

        self.tau     = tau
        self.sigma   = sigma
        self.aggregate = aggregate

        #assert type(tau)==type(sigma)
    
        if isinstance(self.tau, float):
            self.parammode = 'static'
        else:
            self.parammode = 'variable'


        self.sess    = sess
        self.loss    = loss
        self.penalty1 = penalty1
        self.penalty2 = penalty2
        self.dtype   = dtype
        self.devices  = devices
        self.init_var = True if init_var else False
        if isinstance(self.devices, list):
            self.master = self.devices[0]
        else:
            self.master = self.devices
        
        if not isinstance(devices, list):
            self.matmul   = tf.matmul
            self.spmatmul = tf.sparse_tensor_dense_matmul
        else:
            self.matmul   = distops.matmul
            self.spmatmul = distops.spmatmul

        # check shape.
        self.n, self.m = At.shape # changed for sparse A
        self.l, _ = D.shape
        assert (self.n == D.shape[1])

        # setup variables.
        # for A, we need to consider the case where A is larger than 2GB. (to be distributed)
        if isinstance(At, np.ndarray):
        #if not isinstance(At, distmat.DistMat):
            if not isinstance(devices, list):
                with tf.device(self.devices):
                    Ap = tf.placeholder(dtype, shape=At.shape)
                    self.At = variables.Variable(Ap)
                sess.run(self.At.initializer, feed_dict={Ap: At})
                self.A = None
            else:
                self.At = distmat.DistMat.from_dataset(At, devices=self.devices, sess=sess, dtype=self.dtype) 
                self.A = None
            self.A_dense=True
        elif isinstance(At, distmat.DistMat):
            assert all([d1==d2 for d1, d2 in zip(At.devices, self.devices)])
            self.At = At
            self.A_dense=True
        elif isinstance(At, spmatrix):
            if isinstance(D, coo_matrix):
                pass
            else:
                At = At.tocoo()
            if not isinstance(devices, list):
                At_tensor = coo_to_sparsetensor(At)
                At_sorted_op = sparse_ops.sparse_reorder(At_tensor)
                A_tensor = coo_to_sparsetensor(At.T.tocsr().tocoo())
                A_sorted_op  = sparse_ops.sparse_reorder(A_tensor)
                A_sorted, At_sorted = sess.run([A_sorted_op, At_sorted_op])
                with tf.device(self.devices):
                    self.A = sparse_tensor.SparseTensor.from_value(A_sorted)
                    self.At = sparse_tensor.SparseTensor.from_value(At_sorted)
        
                    # b is a constant. (to duplicate)
            else:
                self.At = distmat.DistSpMat.from_spmatrix(At, devices_r=self.devices)
                self.A = None
            self.A_dense=False
        elif isinstance(At, distmat.DistSpMat):
            assert all([d1==d2 for d1, d2 in zip(At.devices_r, self.devices)])
            self.At  = At
            self.A = None
            self.A_dense=False
        else:
            raise ValueError("invalid input for At")
        

        if not isinstance(D, distmat.DistSpMat):

            # D should be COO format.
            # dtype induced from original matrix.
            # avoid recomputation of transpose.
            if isinstance(D, coo_matrix):
                pass
            elif isinstance(D, spmatrix):
                D = D.tocoo()
            else:
                raise ValueError("must be a scipy sparse matrix")
            if not isinstance(devices, list):
                D_tensor = coo_to_sparsetensor(D)
                D_sorted_op = sparse_ops.sparse_reorder(D_tensor)
                Dt_tensor = coo_to_sparsetensor(D.T.tocsr().tocoo())
                Dt_sorted_op = sparse_ops.sparse_reorder(Dt_tensor)
                #Dt_sorted_op = sparse_ops.sparse_reorder(sparse_ops.sparse_transpose(D_tensor))
                D_sorted, Dt_sorted = sess.run([D_sorted_op, Dt_sorted_op])
                with tf.device(self.devices):
                    self.D = sparse_tensor.SparseTensor.from_value(D_sorted)
                    self.Dt = sparse_tensor.SparseTensor.from_value(Dt_sorted)
        
                    # b is a constant. (to duplicate)
            else:
                self.D = distmat.DistSpMat.from_spmatrix(D, devices_r=self.devices)
                self.Dt = None
        else:
            assert all([d1==d2 for d1, d2 in zip(D.devices_r, self.devices)])
            self.D  = D
            self.Dt = None
        

        with tf.device(self.master):
            self.b = constant_op.constant(b, dtype=dtype)

            self._setup_variables(init_var)
            self._setup_evals()
        

    def matmul_A(self, x, transpose_A = False):
        if self.A_dense:
            if self.A:
                return self.matmul(self.A, x, transpose_A)
            else:
                return self.matmul(self.At, x, not transpose_A, master=self.master)
        else:
            if transpose_A:
                r =  self.spmatmul(self.At, x)
                return r
            else:
                if self.A:
                    r = self.spmatmul(self.A, x)
                    #r = tf.sparse_tensor_dense_matmul(self.A, x)
                    #print(x.shape)
                    #print(r.shape)

                    return r
                else:
                    r = self.spmatmul(self.At, x, True)  
                    return r
            

    def spmatmul_D(self, x, transpose_A = False):
        if not transpose_A: 
            return self.spmatmul(self.D, x)
        else:
            if self.Dt:
                return self.spmatmul(self.Dt, x)
            else:
                return self.spmatmul(self.D, x, True)


    def _setup_variables(self, init_var=None):
        if init_var:
            self.x1, self.x2, self.x1ag, self.x2ag, self.y1, self.y2, self.y1ag, self.y2ag = init_var
            self.variables = init_var
        else:
            if not isinstance(self.devices, list):
                with tf.device(self.master):
                    self.x1 = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.x2 = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.x1ag = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.x2ag = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.y1 = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.y2 = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.y1ag = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.y2ag = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
            else:
                self.x1, self.x2 = [], []
                self.x1ag, self.x2ag = [], []
                self.y1, self.y2 = [], []
                self.y1ag, self.y2ag = [], []
                partition_c = self.D.partition_c # partitions n
                partition_r = self.D.partition_r # partitions l
                for i in range(len(self.devices)):
                    with tf.device(self.devices[i]):
                        self.x1.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.x1ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.x2.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.x2ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.y2.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.y2ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.y1.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.y1ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
            self.variables = [self.x1, self.x2, self.x1ag, self.x2ag, self.y1, self.y2, self.y1ag, self.y2ag]
        
    def _setup_evals(self):
        # default obj/resid
        self.default_obj      = constant_op.constant(0, dtype=self.dtype)
        self.default_x1diff = constant_op.constant(0, dtype=self.dtype)
        self.default_x2diff = constant_op.constant(0, dtype=self.dtype)
        self.default_y1diff = constant_op.constant(0, dtype=self.dtype)
        self.default_y2diff = constant_op.constant(0, dtype=self.dtype)
        self.default_x1norm = constant_op.constant(0, dtype=self.dtype)
        self.default_x2norm = constant_op.constant(0, dtype=self.dtype)
        self.default_y1norm = constant_op.constant(0, dtype=self.dtype)
        self.default_y2norm = constant_op.constant(0, dtype=self.dtype)
        self.default_ind_diff = constant_op.constant(0, dtype=self.dtype)
        self.default_evals = [self.default_obj, self.default_x1diff, self.default_x2diff, 
            self.default_y1diff, self.default_y2diff, self.default_x1norm, self.default_x2norm, self.default_y1norm, self.default_y2norm, self.default_ind_diff]

    def _iterate(self, *args):
        raise NotImplementedError

    def _evaluate(self, *args):
        need_eval, x1, x2, x1p, x2p, y1, y2, y1p, y2p = args
        A = self.A
        D = self.D

        #Ax1p   = self.matmul_A(x1p)
        Dtx2p = self.spmatmul_D(x2p, True)
        if isinstance(self.At, distmat.DistSpMat):
            ADtx2p = self.matmul_A(Dtx2p).tensors[0]
        else:
            ADtx2p = self.matmul_A(Dtx2p)
        #Dx2p   = self.spmatmul_D(x2p)
        def calculate_evals():
            if not isinstance(x1p, distmat.DistMat):
                x1_diff = tf.reduce_max(tf.abs(x1p-x1))
                x2_diff = tf.reduce_max(tf.abs(x2p-x2))
                y1_diff = tf.reduce_max(tf.abs(y1p-y1))
                y2_diff = tf.reduce_max(tf.abs(y2p-y2))
                x1_norm = linalg_ops.norm(x1p)
                x2_norm = linalg_ops.norm(x2p)
                y1_norm = linalg_ops.norm(y1p)
                y2_norm = linalg_ops.norm(y2p)
                ind_diff = linalg_ops.norm(x1p - Dtx2p)
            else:
                x1_diff = tf.sqrt(tf.reduce_sum((((x1p-x1)**2).apply(tf.reduce_sum)).tensors))
                x2_diff = tf.sqrt(tf.reduce_sum((((x2p-x2)**2).apply(tf.reduce_sum)).tensors))
                y1_diff = tf.sqrt(tf.reduce_sum((((y1p-y1)**2).apply(tf.reduce_sum)).tensors))
                y2_diff = tf.sqrt(tf.reduce_sum((((y2p-y2)**2).apply(tf.reduce_sum)).tensors))
                x1_norm = tf.sqrt(tf.reduce_sum((((x1p)**2).apply(tf.reduce_sum)).tensors))
                x2_norm = tf.sqrt(tf.reduce_sum((((x2p)**2).apply(tf.reduce_sum)).tensors))
                y1_norm = tf.sqrt(tf.reduce_sum((((y1p)**2).apply(tf.reduce_sum)).tensors))
                y2_norm = tf.sqrt(tf.reduce_sum((((y2p)**2).apply(tf.reduce_sum)).tensors))
                ind_diff = tf.sqrt(tf.reduce_sum((((x1p-Dtx2p)**2).apply(tf.reduce_sum)).tensors))

            objval = self.loss.eval(ADtx2p, self.b) + self.penalty1.eval(x2p) # hard to evaluate objval for latent lasso ### self.loss.eval(Axp, self.b) + self.penalty.eval(Dxp)
            return [objval, x1_diff, x2_diff, y1_diff, y2_diff, x1_norm, x2_norm, y1_norm, y2_norm, ind_diff]
        evals = control_flow_ops.cond(
            need_eval, 
            calculate_evals,
            lambda: self.default_evals)
        return evals

    def solve(self, max_iters=1000, check_interval=10, verbose=False, 
            atol=1e-4, rtol=1e-2, profile=False, output=None, outfile=None, return_var=False):
        t_start = time.time()
        tol = (rtol, atol)
        sess = self.sess
        
        cycle_count = tf.placeholder(tf.int32, shape=tuple())

        # setup evaluation cycle ops and initialization ops
        def cond(k, varz, evals):
            return k < check_interval
        def body(k, varz, evals):
            need_eval = math_ops.equal(k, check_interval-1)
            iter_count = cycle_count*check_interval+k
            varzp, evalsp = self._iterate(varz, tol, need_eval, k, iter_count)
            return [k+1, varzp, evalsp]

        with tf.device(self.master):
            loop_vars = [0, self.variables, self.default_evals]
            _, varz_cycle, evals_cycle = control_flow_ops.while_loop(
                cond, body, loop_vars)
            #print(varz_cycle)
            #print(evals_cycle)
            if not self.init_var:
                if isinstance(self.devices, list):
                    init_op = [variables.variables_initializer(v) for v in self.variables]
                else:
                    init_op = variables.variables_initializer(self.variables)
                # not using global_variables_initializer: A is already initialized!
            else: 
                init_op = []
                #init_op = [variables.variables_initializer(self.xag)]+[variables.variables_initializer(self.yag)]#+[variables.variables_initializer(self.y)]+[variables.variables_initializer(self.ym)]
                
                # this was active before... is it necessary?
                #init_op += [xm.assign(x) for (xm, x) in zip(self.xm, self.x)]  
                #init_op += [ym.assign(y) for (ym, y) in zip(self.ym, self.y)] 

                #init_op += [variables.variables_initializer(self.x)]
                #init_op += [variables.variables_initializer(self.xm)]
                #init_op += [variables.variables_initializer(self.y)]
                #init_op += [variables.variables_initializer(self.ym)]
                #init_op += [variables.variables_initializer(self.xag)]+[variables.variables_initializer(self.yag)]
            if isinstance(self.devices, list):
                cycle_op = [control_flow_ops.group(
                    *[x.assign(val) for x, val in zip(vv, vz)]) for vv, vz in zip(self.variables,
                        varz_cycle)] 
            else:
                cycle_op = control_flow_ops.group(
                    *[x.assign(val) for x, val in zip(self.variables, varz_cycle)])

        if verbose:
            print("Starting solve...")
            print("n={}, p={}, l={}".format(self.m, self.n, self.l))
            print("rtol=%.2e, atol=%.2e" %(rtol, atol))
            print("%5s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %6s" % (
                ("iter", "obj", "x1 diff", "x2 diff", "y1 diff", "y2 diff",  "x1 norm", "x2 norm",  "y1 norm", "y2 norm", "ind_diff",  "time")))
            print("-"*80)
        if init_op:
            sess.run(init_op)
        num_cycles = max_iters // check_interval
        if outfile:
            outd = {}
            outd['iter'] = np.zeros((num_cycles,))
            outd['obj'] = np.zeros((num_cycles,))
            outd['x1diff'] = np.zeros((num_cycles,))
            outd['x2diff'] = np.zeros((num_cycles,))
            outd['y1diff'] = np.zeros((num_cycles,))
            outd['y2diff'] = np.zeros((num_cycles,))
            outd['x1norm'] = np.zeros((num_cycles,))
            outd['x2norm'] = np.zeros((num_cycles,))
            outd['y1norm'] = np.zeros((num_cycles,))
            outd['y2norm'] = np.zeros((num_cycles,))
            outd['ind_diff'] = np.zeros((num_cycles,))
            outd['time'] = np.zeros((num_cycles,))
        for i in range(num_cycles):
            t0 = time.time()
            if profile:
                raise NotImplementedError
                run_options = config_pb2.RunOptions(
                    trace_level=config_pb2.RunOptions.FULL_TRACE)                                     
                run_metadata = config_pb2.RunMetadata()
                _, obj, xdiff, ydiff = sess.run(                                  
                    [cycle_op] + evals_cycle,                                                     
                    options=run_options,
                    run_metadata=run_metadata, feed_dict={i:i})                                                        
                
                tl = timeline.Timeline(run_metadata.step_stats)                                       
                ctf = tl.generate_chrome_trace_format()
                with open("/tmp/prox_pd_cycle_%d.json" % i, "w") as f:                      
                    f.write(ctf)
            else:
                _, obj, x1diff, x2diff, y1diff, y2diff, x1norm, x2norm, y1norm, y2norm, ind_diff = sess.run(
                    [cycle_op]+ evals_cycle, feed_dict={cycle_count:i})
            t1 = time.time()

            if verbose:
                total_iters = (i+1)*check_interval
                print("%5d %20.9e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %7.3f" %
                        (total_iters, obj, x1diff, x2diff, y1diff,y2diff,  x1norm, x2norm, y1norm, y2norm, ind_diff, t1-t0))
            if outfile:
                outd['iter'][i] = total_iters
                outd['obj'][i] = obj
                outd['x1diff'][i] = x1diff
                outd['x2diff'][i] = x2diff
                outd['y1diff'][i] = y1diff
                outd['y2diff'][i] = y2diff
                outd['x1norm'][i] = x1norm
                outd['x2norm'][i] = x2norm
                outd['y1norm'][i] = y1norm
                outd['y2norm'][i] = y2norm
                outd['ind_diff'][i] = ind_diff
                outd['time'][i] = t1-t0
            #if r_norm0 < eps_pri0 and s_norm0 < eps_dual0:
                #break
            # TODO: termination condition
        if outfile:
            scipy.io.savemat(outfile, outd)

        if verbose:
            print("-"*80)
            if i < num_cycles -1:
                status = "Converged"
            else:
                status = "Max iterations"
            print("%s, %.2f seconds."% (status, time.time()-t_start)) 

        if output is None:
            output = self.variables
        val = sess.run(output)
        if return_var:
            for o,v in zip(output, val):
                sess.run(o.assign(v))

            return output
        else: 
            return val
     
        #return sess.run(output)

                    
class Split22WithPrevStep(Split22):
    """
    Primal-dual algorithms with previous step needed for computation
    """
    def _setup_variables(self, init_var=False):
        if init_var:
            ( self.x1m, self.x2m,  self.x1, self.x2, self.x1ag, self.x2ag,
              self.y1m, self.y2m,  self.y1, self.y2, self.y1ag, self.y2ag ) = init_var
            self.variables = init_var
        else:
            if not isinstance(self.devices, list):
                with tf.device(self.master):
                    self.x1 = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.x1m = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.x1ag = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.y1 = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.y1m = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.y1ag = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.x2 = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.x2m = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.x2ag = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.y2 = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.y2m = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.y2ag = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
            else:
                self.x1  = []
                self.x1m = []
                self.x1ag= []
                self.y1  = []
                self.y1m = []
                self.y1ag= []
                self.x2  = []
                self.x2m = []
                self.x2ag= []
                self.y2  = []
                self.y2m = []
                self.y2ag= []
                partition_c = self.D.partition_c # splits n
                partition_r = self.D.partition_r # splits l
                for i in range(len(self.devices)):
                    with tf.device(self.devices[i]):
                        self.x1.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.x1m.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.x1ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.x2.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.x2m.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.x2ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.y1.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.y1m.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.y1ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.y2.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.y2m.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.y2ag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
        
            self.variables = [self.x1m, self.x2m, self.x1, self.x2, self.x1ag, self.x2ag,  
                              self.y1m, self.y2m, self.y1, self.y2, self.y1ag, self.y2ag]


class BaseKappa(Split22):
    def __init__(self, loss, penalty1, penalty2, A, D, b, tau=None, sigma=None, kappa=0., rho=1.0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.kappa = kappa
        self.rho = rho
        super().__init__(loss, penalty1, penalty2, A, D, b, tau, sigma, sess, dtype, devices, init_var=None, aggregate=aggregate)
        
    def _iterate(self, *args):
        (x1, x2, x1ag, x2ag,  y1, y2, y1ag, y2ag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x1, list):
            x1 = distmat.DistMat(x1)
            x2 = distmat.DistMat(x2)
            x1ag = distmat.DistMat(x1ag)
            x2ag = distmat.DistMat(x2ag)
        if isinstance(y1, list):
            y1 = distmat.DistMat(y1)
            y2 = distmat.DistMat(y2)
            y1ag = distmat.DistMat(y1ag)
            y2ag = distmat.DistMat(y2ag)

        with ops.name_scope(type(self).__name__):
            with ops.name_scope("xbar_update"):
                Ax   = self.matmul_A(x1)
                #tf.reshape(Ax, ())
                r    = self.loss.eval_deriv(Ax, self.b)
                print(r.shape)
                Atr  = self.matmul_A(r, True)
                
                Dy2  = self.spmatmul_D(y2)
               
                x1bar = x1 - self.tau * (1-self.kappa) * (Atr + y2)
                x2bar = x2 - self.tau * (1-self.kappa) * (y1 - Dy2)
 
            with ops.name_scope("y_update"):
                
                Dtx2bar = self.spmatmul_D(x2bar, True)

                u1 = x2bar
                u2 = x1bar - Dtx2bar

                y1pp   = self.penalty1.prox(y1 + self.sigma * u1, self.sigma)
                y2pp   = self.penalty2.prox(y2 + self.sigma * u2, self.sigma)
            with ops.name_scope("x_update"):
                y1bar    = (1 + self.kappa) * y1pp - self.kappa * y1
                y2bar    = (1 + self.kappa) * y2pp - self.kappa * y2
                Dy2bar   = self.spmatmul_D(y2bar)

                x1pp   = x1 - self.tau * (Atr + y2bar)
                x2pp   = x2 - self.tau * (y1bar - Dy2bar)
            with ops.name_scope("relax"):
                x1p = (one-self.rho)*x1 + self.rho*x1pp
                x2p = (one-self.rho)*x2 + self.rho*x2pp
                y1p = (one-self.rho)*y1 + self.rho*y1pp
                y2p = (one-self.rho)*y2 + self.rho*y2pp
            with ops.name_scope("aggregate"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    x1agp = (one-one/(iter_num_f+one))*x1ag + 1/(iter_num_f+one)*x1
                    x2agp = (one-one/(iter_num_f+one))*x2ag + 1/(iter_num_f+one)*x2
                    y1agp = (one-one/(iter_num_f+one))*y1ag + 1/(iter_num_f+one)*y1
                    y2agp = (one-one/(iter_num_f+one))*y2ag + 1/(iter_num_f+one)*y2
                else:
                    x1agp = x1ag
                    x2agp = x2ag
                    y1agp = y1ag
                    y2agp = y2ag
            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, x1ag,x2ag, x1agp, x2agp, y1ag,y2ag,  y1agp, y2agp)
                else:
                    evals = self._evaluate(need_eval, x1, x2, x1p, x2p, y1, y2, y1p,y2p)
        if isinstance(x1p, distmat.DistMat):
            x1p = x1p.tensors
            x2p = x2p.tensors
            x1agp = x1agp.tensors
            x2agp = x2agp.tensors
        if isinstance(y1p, distmat.DistMat):
            y1p = y1p.tensors
            y2p = y2p.tensors
            y1agp = y1agp.tensors
            y2agp = y2agp.tensors
        return [x1p,x2p, x1agp,x2agp,  y1p, y2p, y1agp,y2agp], evals 

class BaseLV(Split22):
    def __init__(self, loss, penalty1, penalty2, A, D, b, tau=None, sigma=None, rho=1.0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.rho = rho
        super().__init__(loss, penalty1, penalty2, A, D, b, tau, sigma, sess, dtype, devices, init_var=None, aggregate=aggregate)
        
    def _iterate(self, *args):
        (x1, x2, x1ag, x2ag,  y1, y2, y1ag, y2ag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x1, list):
            x1 = distmat.DistMat(x1)
            x2 = distmat.DistMat(x2)
            x1ag = distmat.DistMat(x1ag)
            x2ag = distmat.DistMat(x2ag)
        if isinstance(y1, list):
            y1 = distmat.DistMat(y1)
            y2 = distmat.DistMat(y2)
            y1ag = distmat.DistMat(y1ag)
            y2ag = distmat.DistMat(y2ag)

        with ops.name_scope(type(self).__name__):


            with ops.name_scope("xbar_update"):
                Ax   = self.matmul_A(x1)
                r    = self.loss.eval_deriv(Ax, self.b)
                Atr  = self.matmul_A(r, True)
                
                Dy2  = self.spmatmul_D(y2)
               
                x1bar = x1 - self.tau * (Atr + y2)
                x2bar = x2 - self.tau * (y1 - Dy2)
 
            with ops.name_scope("y_update"):
                
                Dtx2bar = self.spmatmul_D(x2bar, True)

                u1 = x2bar
                u2 = x1bar - Dtx2bar

                y1pp   = self.penalty1.prox(y1 + self.sigma * u1, self.sigma)
                y2pp   = self.penalty2.prox(y2 + self.sigma * u2, self.sigma)
            with ops.name_scope("x_update"):
                y1bar    =  y1pp 
                y2bar    =  y2pp 
                Dy2bar   = self.spmatmul_D(y2bar)

                x1pp   = x1 - self.tau * (Atr + y2bar)
                x2pp   = x2 - self.tau * (y1bar - Dy2bar)

            with ops.name_scope("relax"):
                x1p = (one-self.rho)*x1 + self.rho*x1pp
                x2p = (one-self.rho)*x2 + self.rho*x2pp
                y1p = (one-self.rho)*y1 + self.rho*y1pp
                y2p = (one-self.rho)*y2 + self.rho*y2pp
            with ops.name_scope("aggregate"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    x1agp = (one-one/(iter_num_f+one))*x1ag + 1/(iter_num_f+one)*x1
                    x2agp = (one-one/(iter_num_f+one))*x2ag + 1/(iter_num_f+one)*x2
                    y1agp = (one-one/(iter_num_f+one))*y1ag + 1/(iter_num_f+one)*y1
                    y2agp = (one-one/(iter_num_f+one))*y2ag + 1/(iter_num_f+one)*y2
                else:
                    x1agp = x1ag
                    x2agp = x2ag
                    y1agp = y1ag
                    y2agp = y2ag
            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, x1ag,x2ag, x1agp, x2agp, y1ag,y2ag,  y1agp, y2agp)
                else:
                    evals = self._evaluate(need_eval, x1, x2, x1p, x2p, y1, y2, y1p,y2p)
        if isinstance(x1p, distmat.DistMat):
            x1p = x1p.tensors
            x2p = x2p.tensors
            x1agp = x1agp.tensors
            x2agp = x2agp.tensors
        if isinstance(y1p, distmat.DistMat):
            y1p = y1p.tensors
            y2p = y2p.tensors
            y1agp = y1agp.tensors
            y2agp = y2agp.tensors
        return [x1p,x2p, x1agp,x2agp,  y1p, y2p, y1agp,y2agp], evals 

class BaseCV(Split22):
    def __init__(self, loss, penalty1, penalty2, A, D, b, tau=None, sigma=None, rho=1.0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.rho = rho
        super().__init__(loss, penalty1, penalty2, A, D, b, tau, sigma, sess, dtype, devices, init_var=None, aggregate=aggregate)
        
    def _iterate(self, *args):
        (x1, x2, x1ag, x2ag,  y1, y2, y1ag, y2ag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x1, list):
            x1 = distmat.DistMat(x1)
            x2 = distmat.DistMat(x2)
            x1ag = distmat.DistMat(x1ag)
            x2ag = distmat.DistMat(x2ag)
        if isinstance(y1, list):
            y1 = distmat.DistMat(y1)
            y2 = distmat.DistMat(y2)
            y1ag = distmat.DistMat(y1ag)
            y2ag = distmat.DistMat(y2ag)

        with ops.name_scope(type(self).__name__):
            with ops.name_scope("x_update"):
                Ax   = self.matmul_A(x1)
                r    = self.loss.eval_deriv(Ax, self.b)
                Atr  = self.matmul_A(r, True)
                
                Dy2  = self.spmatmul_D(y2)
                x1pp = x1 - self.tau * (Atr + y2)
                x2pp = x2 - self.tau * (y1 - Dy2)
            with ops.name_scope("xbar_update"):  
                x1bar = 2 * x1pp - x1
                x2bar = 2 * x2pp - x2 
 
            with ops.name_scope("y_update"):
                
                Dtx2bar = self.spmatmul_D(x2bar, True)

                u1 = x2bar
                u2 = x1bar - Dtx2bar

                y1pp   = self.penalty1.prox(y1 + self.sigma * u1, self.sigma)
                y2pp   = self.penalty2.prox(y2 + self.sigma * u2, self.sigma)
            with ops.name_scope("relax"):
                x1p = (one-self.rho)*x1 + self.rho*x1pp
                x2p = (one-self.rho)*x2 + self.rho*x2pp
                y1p = (one-self.rho)*y1 + self.rho*y1pp
                y2p = (one-self.rho)*y2 + self.rho*y2pp
            with ops.name_scope("aggregate"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    x1agp = (one-one/(iter_num_f+one))*x1ag + 1/(iter_num_f+one)*x1
                    x2agp = (one-one/(iter_num_f+one))*x2ag + 1/(iter_num_f+one)*x2
                    y1agp = (one-one/(iter_num_f+one))*y1ag + 1/(iter_num_f+one)*y1
                    y2agp = (one-one/(iter_num_f+one))*y2ag + 1/(iter_num_f+one)*y2
                else:
                    x1agp = x1ag
                    x2agp = x2ag
                    y1agp = y1ag
                    y2agp = y2ag
            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, x1ag,x2ag, x1agp, x2agp, y1ag,y2ag,  y1agp, y2agp)
                else:
                    evals = self._evaluate(need_eval, x1, x2, x1p, x2p, y1, y2, y1p,y2p)
        if isinstance(x1p, distmat.DistMat):
            x1p = x1p.tensors
            x2p = x2p.tensors
            x1agp = x1agp.tensors
            x2agp = x2agp.tensors
        if isinstance(y1p, distmat.DistMat):
            y1p = y1p.tensors
            y2p = y2p.tensors
            y1agp = y1agp.tensors
            y2agp = y2agp.tensors
        return [x1p,x2p, x1agp,x2agp,  y1p, y2p, y1agp,y2agp], evals 


class OptimalAB(Split22WithPrevStep): #TODO
    def __init__(self, loss, penalty1, penalty2, At, D, b, tau=None, sigma=None, rho=None, theta=None, coef_a=-1, coef_b=0, tau0=0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.rho    = rho
        self.theta  = theta
        self.coef_a = coef_a
        self.coef_b = coef_b
        self.taum = tau0
        super().__init__(loss, penalty1, penalty2, At, D, b, tau, sigma, sess, dtype, devices, aggregate, init_var)
        assert self.parammode == 'variable'

    
    def _iterate(self, *args):
        (x1m, x2m, x1, x2, x1ag, x2ag, y1m, y2m,  y1, y2, y1ag, y2ag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x1, list):
            x1m = distmat.DistMat(x1m)
            x2m = distmat.DistMat(x2m)
            x1 = distmat.DistMat(x1)
            x2 = distmat.DistMat(x2)
            x1ag = distmat.DistMat(x1ag)
            x2ag = distmat.DistMat(x2ag)
        if isinstance(y1, list):
            y1m = distmat.DistMat(y1m)
            y2m = distmat.DistMat(y2m)
            y1 = distmat.DistMat(y1)
            y2 = distmat.DistMat(y2)
            y1ag = distmat.DistMat(y1ag)
            y2ag = distmat.DistMat(y2ag)

        # 1-based indexing for params
        taum  = tf.to_float(self.tau(iter_num))
        tau   = tf.to_float(self.tau(iter_num+1))
        sigma = tf.to_float(self.sigma(iter_num+1))
        theta = tf.to_float(self.theta(iter_num+1))
        rho   = tf.to_float(self.rho(iter_num+1))
        coef_a = self.coef_a
        coef_b = self.coef_b
        with ops.name_scope(type(self).__name__):
            with ops.name_scope("xmid_update"):
                u1 = (1 - theta * coef_a) * x2 + theta * coef_a * x2m
                Dtu1 = self.spmatmul_D(u1, True)
                u2 = (1 - theta * coef_a) * x1 + theta * coef_a * x1m - Dtu1

                c = theta * (taum/tau*(1+coef_b)-coef_b)
                v1 = (1+c) * y2 - c * y2m
                Dv1 = self.spmatmul_D(v1)
                v2 = (1+c) * y1 - c * y1m - Dv1
                 
                x1mid  = (1 - rho) * x1ag + rho * x1
                Ax1mid = self.matmul_A(x1mid)

            with ops.name_scope("update_iterates"):
                r     = self.loss.eval_deriv(Ax1mid, self.b)
                Atr   = self.matmul_A(r, True)

                Dtv2 = self.spmatmul_D(v2, True)
                
                u1p   = u1 - tau * (1+coef_a) * v2
                u2p   = u2 - tau * (1+coef_a) * (Atr + v1 - Dtv2)

                y1p = self.penalty1.prox(y1 + sigma * u1p, sigma)
                y2p = self.penalty2.prox(y2 + sigma * u2p, sigma)

                v1p = (1+coef_b) * y2p - coef_b *(1+theta) * y2 + theta * coef_b * y2m
                Dv1p = self.spmatmul_D(v1p)
                v2p = (1+coef_b) * y1p - coef_b *(1+theta) * y1 + theta * coef_b * y1m - Dv1p

                         
                x1p = x1 - tau * (Atr + v1p)
                x2p = x2 - tau * v2p 


            with ops.name_scope("aggregation"):
                x1agp  = (1-rho) * x1ag + rho * x1p
                x2agp  = (1-rho) * x2ag + rho * x2p
                y1agp  = (1-rho) * y1ag + rho * y1p
                y2agp  = (1-rho) * y2ag + rho * y2p

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, x1ag, x2ag, x1agp, x2agp, y1ag, y2ag, y1agp, y2agp)
                else:
                    evals = self._evaluate(need_eval, x1, x2, x1p, x2p, y1, y2, y1p, y2p)
                
        if isinstance(x1p, distmat.DistMat):
            x1 = x1.tensors
            x2 = x2.tensors
            x1p = x1p.tensors
            x2p = x2p.tensors
            x1agp = x1agp.tensors
            x2agp = x2agp.tensors
        if isinstance(y1p, distmat.DistMat):
            y1 = y1.tensors
            y2 = y2.tensors
            y1p = y1p.tensors
            y2p = y2p.tensors
            y1agp = y1agp.tensors
            y2agp = y2agp.tensors
        return [x1, x2, x1p, x2p, x1agp, x2agp, y1, y2, y1p, y2p, y1agp, y2agp], evals

class FBFInertial(Split22WithPrevStep): 
    def __init__(self, loss, penalty1, penalty2, At, D, b, tau=None, alpha1=0.01, alpha2=0.01,  sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        super().__init__(loss, penalty1, penalty2, At, D, b, tau, 1.0, sess, dtype, devices, aggregate, init_var)
        #assert self.parammode == 'variable'

    
    def _iterate(self, *args):
        (x1m, x2m, x1, x2, x1ag, x2ag, y1m, y2m,  y1, y2, y1ag, y2ag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x1, list):
            x1m = distmat.DistMat(x1m)
            x2m = distmat.DistMat(x2m)
            x1 = distmat.DistMat(x1)
            x2 = distmat.DistMat(x2)
            x1ag = distmat.DistMat(x1ag)
            x2ag = distmat.DistMat(x2ag)
        if isinstance(y1, list):
            y1m = distmat.DistMat(y1m)
            y2m = distmat.DistMat(y2m)
            y1 = distmat.DistMat(y1)
            y2 = distmat.DistMat(y2)
            y1ag = distmat.DistMat(y1ag)
            y2ag = distmat.DistMat(y2ag)

        tau = self.tau
        alpha1 = self.alpha1
        alpha2 = self.alpha2

        with ops.name_scope(type(self).__name__):
            with ops.name_scope("x_update"):
                Ax1 = self.matmul_A(x1)
                r     = self.loss.eval_deriv(Ax1, self.b)
                Atr   = self.matmul_A(r, True)

                Dy2 = self.spmatmul_D(y2)
 
                x1tilde = x1 - tau * (Atr + y2) + alpha1 * (x1 - x1m)
                x2tilde = x2 - tau * (y1 - Dy2) + alpha1 * (x2 - x2m)


            with ops.name_scope("y_update"):
                Dtx2 = self.spmatmul_D(x2, True)
                
                y1tilde = self.penalty1.prox(y1 + tau * x2 + alpha1 * (y1 - y1m), self.penalty1.lam)
                y2tilde = self.penalty2.prox(y2 + tau * (x1 - Dtx2) + alpha1 * (y2 - y2m), 1)
            with ops.name_scope("correction"):
                Dtxdiff = self.spmatmul_D(x2tilde - x2, True)
                Dydiff  = self.spmatmul_D(y2 - y2tilde)

                y1p = y1tilde + tau * (x2tilde - x2) + alpha2 *(y1 - y1m)
                y2p = y2tilde + tau * ((x1tilde-x1) - Dtxdiff) + alpha2 * (y2 - y2m)
 
                x1p = x1tilde + tau * (y2 - y2tilde) + alpha2 * (x1 - x1m)
                x2p = x2tilde + tau * ((y1 - y1tilde) - Dydiff) + alpha2 * (x2 - x2m)


            with ops.name_scope("aggregation"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    x1agp = (one-one/(iter_num_f+one))*x1ag + 1/(iter_num_f+one)*x1
                    x2agp = (one-one/(iter_num_f+one))*x2ag + 1/(iter_num_f+one)*x2
                    y1agp = (one-one/(iter_num_f+one))*y1ag + 1/(iter_num_f+one)*y1
                    y2agp = (one-one/(iter_num_f+one))*y2ag + 1/(iter_num_f+one)*y2
                else:
                    x1agp = x1ag
                    x2agp = x2ag
                    y1agp = y1ag
                    y2agp = y2ag

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, x1ag, x2ag, x1agp, x2agp, y1ag, y2ag, y1agp, y2agp)
                else:
                    evals = self._evaluate(need_eval, x1, x2, x1p, x2p, y1, y2, y1p, y2p)
                
        if isinstance(x1p, distmat.DistMat):
            x1 = x1.tensors
            x2 = x2.tensors
            x1p = x1p.tensors
            x2p = x2p.tensors
            x1agp = x1agp.tensors
            x2agp = x2agp.tensors
        if isinstance(y1p, distmat.DistMat):
            y1 = y1.tensors
            y2 = y2.tensors
            y1p = y1p.tensors
            y2p = y2p.tensors
            y1agp = y1agp.tensors
            y2agp = y2agp.tensors
        return [x1, x2, x1p, x2p, x1agp, x2agp, y1, y2, y1p, y2p, y1agp, y2agp], evals




   








   








