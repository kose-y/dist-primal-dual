from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
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
import scipy

one = tf.constant(1.0)

class PrimalDual:
    def __init__(self, loss, penalty, At, D, b, tau=None, sigma=None, sess=None, dtype=dtypes.float32, devices='', aggregate=False, init_var=None):
        if not tau:
            raise ValueError("Must set tau")
        if not sigma:
            raise ValueError("Must set sigma")
        if sess is None:
            sess = session.Session()

        self.tau     = tau
        self.sigma   = sigma
        self.aggregate = aggregate

        #print(type(tau), type(sigma))

        #assert type(tau)==type(sigma)
    
        if isinstance(self.tau, float):
            self.parammode = 'static'
        else:
            self.parammode = 'variable'


        self.sess    = sess
        self.loss    = loss
        self.penalty = penalty
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
        self.m, self.n = At.T.shape
        self.l, _ = D.shape
        print(At.T.shape)
        print(D.shape)
        print(self.n, D.shape[1])
        assert (self.n == D.shape[1])

        # setup variables.
        # for A, we need to consider the case where A is larger than 2GB. (to be distributed)
        if not isinstance(At, distmat.DistMat):
            if not isinstance(devices, list):
                with tf.device(self.devices):
                    Ap = tf.placeholder(dtype, shape=At.shape)
                    self.At = variables.Variable(Ap)
                sess.run(self.At.initializer, feed_dict={Ap: At})
                self.A = None
            else:
                self.At = distmat.DistMat.from_dataset(At, devices=self.devices, sess=sess, dtype=self.dtype) 
                self.A = None
        else:
            assert all([d1==d2 for d1, d2 in zip(At.devices, self.devices)])
            self.At = At
        

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
                Dt_sorted_op = sparse_ops.sparse_reorder(sparse_ops.sparse_transpose(D_tensor))
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
        if self.A:
            return self.matmul(self.A, x, transpose_A)
        else:
            return self.matmul(self.At, x, not transpose_A, master=self.master)
            

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
            self.x, self.xag, self.y, self.yag = init_var
            self.variables = init_var
        else:
            if not isinstance(self.devices, list):
                with tf.device(self.master):
                    self.x = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.xag = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.y = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.yag = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
            else:
                self.x = []
                self.xag = []
                self.y = []
                self.yag = []
                partition_c = self.D.partition_c
                partition_r = self.D.partition_r
                for i in range(len(self.devices)):
                    with tf.device(self.devices[i]):
                        self.x.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.xag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.y.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.yag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
            self.variables = [self.x, self.xag, self.y, self.yag]
        
    def _setup_evals(self):
        # default obj/resid
        self.default_obj      = constant_op.constant(0, dtype=self.dtype)
        self.default_xdiff = constant_op.constant(0, dtype=self.dtype)
        self.default_ydiff = constant_op.constant(0, dtype=self.dtype)
        self.default_xnorm = constant_op.constant(0, dtype=self.dtype)
        self.default_ynorm = constant_op.constant(0, dtype=self.dtype)
        self.default_evals = [self.default_obj, self.default_xdiff, self.default_ydiff, 
            self.default_xnorm, self.default_ynorm]

    def _iterate(self, *args):
        raise NotImplementedError

    def _evaluate(self, *args):
        need_eval, x, xp, y, yp = args
        A = self.A
        D = self.D

        Axp   = self.matmul_A(xp)
        Dxp   = self.spmatmul_D(xp)
        def calculate_evals():
            if not isinstance(xp, distmat.DistMat):
                x_diff = linalg_ops.norm(xp-x)
                y_diff = linalg_ops.norm(yp-y)
                x_norm = linalg_ops.norm(xp)
                y_norm = linalg_ops.norm(yp)
            else:
                x_diff = tf.sqrt(tf.reduce_sum((((xp-x)**2).apply(tf.reduce_sum)).tensors))
                y_diff = tf.sqrt(tf.reduce_sum((((yp-y)**2).apply(tf.reduce_sum)).tensors))
                x_norm = tf.sqrt(tf.reduce_sum((((xp)**2).apply(tf.reduce_sum)).tensors))
                y_norm = tf.sqrt(tf.reduce_sum((((yp)**2).apply(tf.reduce_sum)).tensors))
            objval = self.loss.eval(Axp, self.b) + self.penalty.eval(Dxp)
            return [objval, x_diff, y_diff, x_norm, y_norm]
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
                init_op += [xm.assign(x) for (xm, x) in zip(self.xm, self.x)]  
                init_op += [ym.assign(y) for (ym, y) in zip(self.ym, self.y)] 
                #init_op += [variables.variables_initializer(self.x)]
                #init_op += [variables.variables_initializer(self.xm)]
                #init_op += [variables.variables_initializer(self.y)]
                #init_op += [variables.variables_initializer(self.ym)]
                init_op += [variables.variables_initializer(self.xag)]+[variables.variables_initializer(self.yag)]
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
            print("%5s %10s %10s %10s %10s %10s %6s" % (
                ("iter", "obj", "x diff", "y diff", "x norm", "y norm",  "time")))
            print("-"*80)
        if init_op:
            sess.run(init_op)
        num_cycles = max_iters // check_interval
        if outfile:
            outd = {}
            outd['iter'] = np.zeros((num_cycles,))
            outd['obj'] = np.zeros((num_cycles,))
            outd['xdiff'] = np.zeros((num_cycles,))
            outd['ydiff'] = np.zeros((num_cycles,))
            outd['xnorm'] = np.zeros((num_cycles,))
            outd['ynorm'] = np.zeros((num_cycles,))
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
                _, obj, xdiff, ydiff, xnorm, ynorm= sess.run(
                    [cycle_op]+ evals_cycle, feed_dict={cycle_count:i})
            t1 = time.time()

            if verbose:
                total_iters = (i+1)*check_interval
                print("%5d %20.9e %10.2e %10.2e %10.2e %10.2e %7.3f" %
                        (total_iters, obj, xdiff, ydiff, xnorm, ynorm, t1-t0))
            if outfile:
                outd['iter'][i] = total_iters
                outd['obj'][i] = obj
                outd['xdiff'][i] = xdiff
                outd['ydiff'][i] = ydiff
                outd['xnorm'][i] = xnorm
                outd['ynorm'][i] = ynorm
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
            return output
        else: 
            return val
     
        #return sess.run(output)

                    
class PrimalDualWithPrevStep(PrimalDual):
    """
    Primal-dual algorithms with previous step needed for computation
    """
    def _setup_variables(self, init_var=False):
        if init_var:
            self.xm, self.x, self.xag, self.ym, self.y, self.yag = init_var
            self.variables = init_var
        else:
            if not isinstance(self.devices, list):
                with tf.device(self.master):
                    self.x = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.xm = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.xag = variables.Variable(
                        array_ops.zeros(shape=(self.n, 1), dtype=self.dtype))
                    self.y = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.ym = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
                    self.yag = variables.Variable(
                        array_ops.zeros(shape=(self.l, 1), dtype=self.dtype))
            else:
                self.x  = []
                self.xm = []
                self.xag= []
                self.y  = []
                self.ym = []
                self.yag= []
                partition_c = self.D.partition_c
                partition_r = self.D.partition_r
                for i in range(len(self.devices)):
                    with tf.device(self.devices[i]):
                        self.x.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.xm.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.xag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_c[i+1]-partition_c[i], 1), dtype=self.dtype)))
                        self.y.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.ym.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
                        self.yag.append(variables.Variable(
                            array_ops.zeros(shape=(partition_r[i+1]-partition_r[i], 1), dtype=self.dtype)))
        
            self.variables = [self.xm, self.x, self.xag, self.ym, self.y, self.yag]


class BaseLV(PrimalDual):
    def _iterate(self, *args):
        # TODO: Check if CSE works (low priority)
        (x, xag, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

        with ops.name_scope(type(self).__name__):
            with ops.name_scope("xbar_update"):
                Ax    = self.matmul_A(x) 
                r     = self.loss.eval_deriv(Ax, self.b)
                Atr   = self.matmul_A(r, True)
                Dty   = self.spmatmul_D(y, True)
                #print (type(Atr))
                #print (type(Dty))
                xbar  = x - self.tau * (Atr+Dty)
            with ops.name_scope("y_update"):
                Dxbar = self.spmatmul_D(xbar)
                yp    = self.penalty.prox(y + self.sigma * Dxbar,self.sigma)
            with ops.name_scope("x_update"):
                Dtyp  = self.spmatmul_D(yp, True)
                xp    = x - self.tau * (Atr + Dtyp)
            with ops.name_scope("aggregate"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    xagp = (one-one/(iter_num_f+one))*xag + 1/(iter_num_f+one)*x
                    yagp = (one-one/(iter_num_f+one))*yag + 1/(iter_num_f+one)*y
                else:
                    xagp = xag
                    yagp = yag
            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
        if isinstance(xp, distmat.DistMat):
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            yp = yp.tensors
            yagp = yagp.tensors
        return [xp, xagp, yp, yagp], evals 


class BaseCV(PrimalDual):
    def _iterate(self, *args):
        (x, xag, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

        with ops.name_scope(type(self).__name__):
            with ops.name_scope("x_update"):
                Ax  = self.matmul_A(x)
                r   = self.loss.eval_deriv(Ax, self.b)
                Atr = self.matmul_A(r, True)
                Dty = self.spmatmul_D(y, True)
                xp = x - self.tau * (Atr+Dty)
            with ops.name_scope("xbar_update"):
                xbar = 2*xp - x
            with ops.name_scope("y_update"):
                Dxbar = self.spmatmul_D(xbar)
                yp    = self.penalty.prox(y + self.sigma * Dxbar, self.sigma)
            with ops.name_scope("aggregate"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    xagp = (one-one/(iter_num_f+one))*xag + 1/(iter_num_f+one)*x
                    yagp = (one-one/(iter_num_f+one))*yag + 1/(iter_num_f+one)*y
                else:
                    xagp = xag
                    yagp = yag
            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
        if isinstance(xp, distmat.DistMat):
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            yp = yp.tensors
            yagp = yagp.tensors
        return [xp, xagp, yp, yagp], evals 

class BaseKappa(PrimalDual):
    def __init__(self, loss, penalty, A, D, b, tau=None, sigma=None, kappa=0., rho=1.0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.kappa = kappa
        self.rho = rho
        super().__init__(loss, penalty, A, D, b, tau, sigma, sess, dtype, devices, init_var=init_var, aggregate=aggregate)
        
    def _iterate(self, *args):
        (x, xag, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

        with ops.name_scope(type(self).__name__):
            with ops.name_scope("xbar_update"):
                Ax   = self.matmul_A(x)
                r    = self.loss.eval_deriv(Ax, self.b)
                Atr  = self.matmul_A(r, True)
                Dty  = self.spmatmul_D(y, True)
                xbar = x - self.tau * (1-self.kappa) * (Atr+Dty)
            with ops.name_scope("y_update"):
                Dxbar = self.spmatmul_D(xbar)
                ypp    = self.penalty.prox(y + self.sigma * Dxbar, self.sigma)
            with ops.name_scope("x_update"):
                ybar  = -self.kappa * y + (1+self.kappa) * ypp
                Dtybar= self.spmatmul_D(ybar, True)
                xpp    = x - self.tau * (Atr + Dtybar)
            with ops.name_scope("relax"):
                xp = (one-self.rho)*x + self.rho*xpp
                yp = (one-self.rho)*y + self.rho*ypp
            with ops.name_scope("aggregate"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    xagp = (one-one/(iter_num_f+one))*xag + 1/(iter_num_f+one)*x
                    yagp = (one-one/(iter_num_f+one))*yag + 1/(iter_num_f+one)*y
                else:
                    xagp = xag
                    yagp = yag
            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
        if isinstance(xp, distmat.DistMat):
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            yp = yp.tensors
            yagp = yagp.tensors
        return [xp, xagp, yp, yagp], evals 
class FBFInertial(PrimalDualWithPrevStep):
    def __init__(self, loss, penalty, A, D, b, tau=None, alpha1=0.01, alpha2=0.01, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.alpha1=alpha1
        self.alpha2=alpha2
        super().__init__(loss, penalty, A, D, b, tau, 1.0, sess, dtype, devices, init_var=init_var, aggregate=aggregate)
    def _iterate(self, *args):
        (xm, x, xag, ym, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            xm = distmat.DistMat(xm)
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            ym = distmat.DistMat(ym)
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)
        tau = self.tau
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        with ops.name_scope(type(self).__name__):
            with ops.name_scope("x_update"):
                Ax  = self.matmul_A(x)
                Dx  = self.spmatmul_D(x)
                Dty = self.spmatmul_D(y, True)

                r   = self.loss.eval_deriv(Ax, self.b)
                Atr = self.matmul_A(r, True)
                xtilde = x - tau * (Atr+Dty)+ alpha1 * (x - xm)
            with ops.name_scope("w_update"):
                Dxtilde = self.spmatmul_D(xtilde)
                ytilde  = self.penalty.prox(y+tau*Dx+alpha1*(y - ym), self.penalty.lam)
                Dtytilde= self.spmatmul_D(ytilde, True)
            with ops.name_scope("correction"):
                yp = ytilde + tau * (Dxtilde - Dx) + alpha2 * (y - ym)
                xp = xtilde + tau *self.spmatmul_D(y-ytilde, True) + alpha2 *(x - xm)
 
            with ops.name_scope("aggregation"):
                if self.aggregate:
                    iter_num_f = tf.to_float(iter_num)
                    xagp = (one-one/(iter_num_f+one))*xag + 1/(iter_num_f+one)*x
                    yagp = (one-one/(iter_num_f+one))*yag + 1/(iter_num_f+one)*y
                else:
                    xagp = xag
                    yagp = yag

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
                
        if isinstance(xp, distmat.DistMat):
            x = x.tensors
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            y = y.tensors
            yp = yp.tensors
            yagp = yagp.tensors
        return [x, xp, xagp, y, yp, yagp], evals

class OptimalAB(PrimalDualWithPrevStep): #TODO
    def __init__(self, loss, penalty, At, D, b, tau=None, sigma=None, rho=None, theta=None, coef_a=-1, coef_b=0, tau0=0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.rho    = rho
        self.theta  = theta
        self.coef_a = coef_a
        self.coef_b = coef_b
        self.taum = tau0
        super().__init__(loss, penalty, At, D, b, tau, sigma, sess, dtype, devices, aggregate, init_var)
        assert self.parammode == 'variable'

    
    def _iterate(self, *args):
        (xm, x, xag, ym, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            xm = distmat.DistMat(xm)
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            ym = distmat.DistMat(ym)
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

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
                Dxm   = self.spmatmul_D(xm)
                Dx    = self.spmatmul_D(x)
                ubar  = Dx - theta * coef_a * (Dx - Dxm)

                Dtym  = self.spmatmul_D(ym, True)
                Dty   = self.spmatmul_D(y, True)
                vbar  = Dty + theta * (taum/tau*(1+coef_b)-coef_b) * (Dty - Dtym)
           
                xmid  = (1 - rho) * xag + rho * x
                Axmid = self.matmul_A(xmid)

            with ops.name_scope("update_iterates"):
                r     = self.loss.eval_deriv(Axmid, self.b)
                Atr   = self.matmul_A(r, True)
                
                up    = ubar - tau*(1+coef_a)* self.spmatmul_D(Atr + vbar)
                yp    = self.penalty.prox(y + sigma * up, sigma)
                Dtyp  = self.spmatmul_D(yp, True)
                vp    = Dtyp + coef_b * (Dtyp - Dty) - theta * coef_b * (Dty - Dtym)
                xp    = x - tau * (Atr + vp)

            with ops.name_scope("aggregation"):
                xagp  = (1-rho) * xag + rho * xp
                yagp  = (1-rho) * yag + rho * yp

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
                
        if isinstance(xp, distmat.DistMat):
            x = x.tensors
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            y = y.tensors
            yp = yp.tensors
            yagp = yagp.tensors
        return [x, xp, xagp, y, yp, yagp], evals


class OptimalABStochastic(OptimalAB):
    def __init__(self, loss, penalty, A_p, D_p, At, D, b, tau=None, sigma=None, rho=None, theta=None, coef_a=-1, coef_b=0, tau0=0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.A_p = float(A_p)
        self.D_p = float(D_p) 
        self.spmatmul_dropout = distops.spmatmul_dropout
        self.matmul_dropout   = distops.matmul_dropout
        super().__init__(loss, penalty, At, D, b, tau, sigma, rho, theta, coef_a, coef_b, tau0, sess, dtype, devices, aggregate, init_var)
        
    def _iterate(self, *args):
        (xm, x, xag, ym, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            xm = distmat.DistMat(xm)
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            ym = distmat.DistMat(ym)
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

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
                
                #Dxm   = self.spmatmul_dropout(self.D, xm, self.D_p)
                Dx1   = self.spmatmul_dropout(self.D, x, self.D_p)
                Dx2   = self.spmatmul_dropout(self.D, x-xm, self.D_p)
                ubar  = Dx1 - theta * coef_a * (Dx2)
                
                Dty1  = self.spmatmul_dropout(self.D, y + theta*taum/tau*(y-ym), self.D_p, True)
                Dty2  = self.spmatmul_dropout(self.D, (taum/tau-1)*(y-ym), self.D_p, True)

                #Dtym  = self.spmatmul_dropout(self.D, ym, self.D_p, True)
                #Dty   = self.spmatmul_dropout(self.D, y, self.D_p, True)
                vbar  = Dty1 + theta * coef_b* Dty2
           
                xmid  = (1 - rho) * xag + rho * x
                Axmid = self.matmul_dropout(self.At, xmid, self.A_p, 'row', True)

            with ops.name_scope("update_iterates"):
                r     = self.loss.eval_deriv(Axmid, self.b)
                Atr   = self.matmul_A(r, True)
                
                up    = ubar - tau*(1+coef_a)* self.spmatmul_D(Atr + vbar)
                yp    = self.penalty.prox(y + sigma * up, sigma)
              
                Dty3  = self.spmatmul_dropout(self.D, yp, self.D_p, True)
                Dty4  = self.spmatmul_dropout(self.D, (yp-y) - theta*(y-ym), self.D_p, True)
                #Dtyp  = self.spmatmul_dropout(self.D, yp, self.D_p, True)
                vp    = Dty3 + coef_b * Dty4
                xp    = x - tau * (Atr + vp)

            with ops.name_scope("aggregation"):
                xagp  = (1-rho) * xag + rho * xp
                yagp  = (1-rho) * yag + rho * yp

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
                
        if isinstance(xp, distmat.DistMat):
            x = x.tensors
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            y = y.tensors
            yp = yp.tensors
            yagp = yagp.tensors
        return [x, xp, xagp, y, yp, yagp], evals


class OptimalABStochastic2(OptimalAB):
    def __init__(self, loss, penalty, A_p, D_p, At, D, b, tau=None, sigma=None, rho=None, theta=None, coef_a=-1, coef_b=0, tau0=0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.A_p = float(A_p)
        self.D_p = float(D_p) 
        super().__init__(loss, penalty, At, D, b, tau, sigma, rho, theta, coef_a, coef_b, tau0, sess, dtype, devices, aggregate, init_var)
        
    def _iterate(self, *args):
        (xm, x, xag, ym, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            xm = distmat.DistMat(xm)
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            ym = distmat.DistMat(ym)
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

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
                
                #Dxm   = self.spmatmul_dropout(self.D, xm, self.D_p)
                Dx1   = self.spmatmul_D(x.dropout(self.D_p))
                Dx2   = self.spmatmul_D( (x-xm).dropout(self.D_p))
                ubar  = Dx1 - theta * coef_a * (Dx2)
                
                Dty1  = self.spmatmul_D( (y + theta*taum/tau*(y-ym)).dropout(self.D_p), True)
                Dty2  = self.spmatmul_D( ((taum/tau-1)*(y-ym)).dropout(self.D_p), True)

                #Dtym  = self.spmatmul_dropout(self.D, ym, self.D_p, True)
                #Dty   = self.spmatmul_dropout(self.D, y, self.D_p, True)
                vbar  = Dty1 + theta * coef_b* Dty2
           
                xmid  = (1 - rho) * xag + rho * x
                Axmid = self.matmul_A( xmid.dropout(self.A_p) )

            with ops.name_scope("update_iterates"):
                r     = self.loss.eval_deriv(Axmid, self.b)
                Atr   = self.matmul_A(r, True)
                
                up    = ubar - tau*(1+coef_a)* self.spmatmul_D(Atr + vbar)
                yp    = self.penalty.prox(y + sigma * up, sigma)
              
                Dty3  = self.spmatmul_D( yp.dropout(self.D_p), True)
                Dty4  = self.spmatmul_D( ((yp-y) - theta*(y-ym)).dropout(self.D_p), True)
                #Dtyp  = self.spmatmul_dropout(self.D, yp, self.D_p, True)
                vp    = Dty3 + coef_b * Dty4
                xp    = x - tau * (Atr + vp)

            with ops.name_scope("aggregation"):
                xagp  = (1-rho) * xag + rho * xp
                yagp  = (1-rho) * yag + rho * yp

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)
                
        if isinstance(xp, distmat.DistMat):
            x = x.tensors
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            y = y.tensors
            yp = yp.tensors
            yagp = yagp.tensors
        return [x, xp, xagp, y, yp, yagp], evals
   








   


class OptimalABStochastic(OptimalAB):
    def __init__(self, loss, penalty, A_p, D_p, At, D, b, tau=None, sigma=None, rho=None, theta=None, coef_a=-1, coef_b=0, tau0=0, sess=None, dtype=dtypes.float32, devices='', aggregate=True, init_var=None):
        self.A_p = float(A_p)
        self.D_p = float(D_p)
        self.spmatmul_dropout = distops.spmatmul_dropout
        self.matmul_dropout   = distops.matmul_dropout
        super().__init__(loss, penalty, At, D, b, tau, sigma, rho, theta, coef_a, coef_b, tau0, sess, dtype, devices, aggregate, init_var)

    def _iterate(self, *args):
        (xm, x, xag, ym, y, yag), (rtol, atol), need_eval, k, iter_num = args
        if isinstance(x, list):
            xm = distmat.DistMat(xm)
            x = distmat.DistMat(x)
            xag = distmat.DistMat(xag)
        if isinstance(y, list):
            ym = distmat.DistMat(ym)
            y = distmat.DistMat(y)
            yag = distmat.DistMat(yag)

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

                #Dxm   = self.spmatmul_dropout(self.D, xm, self.D_p)
                Dx1   = self.spmatmul_dropout(self.D, x, self.D_p)
                Dx2   = self.spmatmul_dropout(self.D, x-xm, self.D_p)
                ubar  = Dx1 - theta * coef_a * (Dx2)

                Dty1  = self.spmatmul_dropout(self.D, y + theta*taum/tau*(y-ym), self.D_p, True)
                Dty2  = self.spmatmul_dropout(self.D, (taum/tau-1)*(y-ym), self.D_p, True)

                #Dtym  = self.spmatmul_dropout(self.D, ym, self.D_p, True)
                #Dty   = self.spmatmul_dropout(self.D, y, self.D_p, True)
                vbar  = Dty1 + theta * coef_b* Dty2

                xmid  = (1 - rho) * xag + rho * x
                Axmid = self.matmul_dropout(self.At, xmid, self.A_p, 'row', True)

            with ops.name_scope("update_iterates"):
                r     = self.loss.eval_deriv(Axmid, self.b)
                Atr   = self.matmul_A(r, True)

                up    = ubar - tau*(1+coef_a)* self.spmatmul_D(Atr + vbar)
                yp    = self.penalty.prox(y + sigma * up, sigma)

                Dty3  = self.spmatmul_dropout(self.D, yp, self.D_p, True)
                Dty4  = self.spmatmul_dropout(self.D, (yp-y) - theta*(y-ym), self.D_p, True)
                #Dtyp  = self.spmatmul_dropout(self.D, yp, self.D_p, True)
                vp    = Dty3 + coef_b * Dty4
                xp    = x - tau * (Atr + vp)

            with ops.name_scope("aggregation"):
                xagp  = (1-rho) * xag + rho * xp
                yagp  = (1-rho) * yag + rho * yp

            with ops.name_scope("evaluations"):
                if self.aggregate:
                    evals = self._evaluate(need_eval, xag, xagp, yag, yagp)
                else:
                    evals = self._evaluate(need_eval, x, xp, y, yp)

        if isinstance(xp, distmat.DistMat):
            x = x.tensors
            xp = xp.tensors
            xagp = xagp.tensors
        if isinstance(yp, distmat.DistMat):
            y = y.tensors
            yp = yp.tensors
            yagp = yagp.tensors
        return [x, xp, xagp, y, yp, yagp], evals
  








