from scipy.sparse import csc_matrix
import numpy as np
import tensorflow as tf
from dist_pd.utils import coo_to_sparsetensor, find_nearest, gidx_to_partition
import dist_pd.partitioners as partitioners
import dist_pd.distmat as distmat
import dist_pd.distops as distops
class PenaltyFunction:
    def __init__(self, name=None):
        self.name = name or type(self).__name__
        with tf.name_scope(self.name):
            with tf.name_scope('initialize'):
                self.initialize()
    def prox(self,y):
        raise NotImplementedError
    def eval(self, y):
        raise NotImplementedError
    def initialize(self):
        pass


class Ind0(PenaltyFunction):
    def __init__(self, name=None):
        super().__init__(name)
    def prox(self, pre_prox, scale):
        if isinstance(pre_prox, distmat.DistMat):
            return pre_prox.apply(self.prox, scale)
        with tf.name_scope(self.name):
            with tf.name_scope("prox"):
                return pre_prox
    def eval(self, Dx):
        with tf.name_scope(self.name):
            with tf.name_scope("eval"):
                if isinstance(Dx, distmat.DistMat):
                    return 0
   

class L1Penalty(PenaltyFunction):
    def __init__(self, lam, name=None):
        super().__init__(name)
        self.lam = lam
    def prox(self, pre_prox, scale):
        if isinstance(pre_prox, distmat.DistMat):
            return pre_prox.apply(self.prox, scale)
        with tf.name_scope(self.name):
            with tf.name_scope("prox"):
                return tf.maximum(tf.minimum(self.lam, pre_prox), -self.lam)
    def eval(self, Dx):
        with tf.name_scope(self.name):
            with tf.name_scope("eval"):
                if isinstance(Dx, distmat.DistMat):
                    absDx = Dx.apply(tf.abs)
                    reduce_sum = tf.reduce_sum(absDx.apply(tf.reduce_sum).tensors)
                    return self.lam*reduce_sum
                else:
                    return self.lam * tf.reduce_sum(tf.abs(Dx))
    def eval_deriv(self, Dx):
        with tf.name_scope(self.name):
            with tf.name_scope("eval_deriv"):
                if isinstance(Dx, distmat, DistMat):
                    sgnDx = Dx.apply(tf.sign)

                    return self.lam * sgnDx
                else:
                    return self.lam * tf.sign(Dx)


# this is based on sparse matrix multiplication
class GroupLasso(PenaltyFunction):
    def __init__(self, lam, g, devices='', partition=None, name=None, dtype=tf.float32):
        assert all([g[i+1]>=g[i] for i in range(len(g)-1)])
        self.lam = lam
        self.g = g.flatten()
        self.gpart = gidx_to_partition(self.g)
        self.dtype=dtype
        self.devices=devices
        self.partition = partition
        super().__init__(name)

    def initialize(self):
        #g_int16 = self.g.astype('int16')
        #sizes = np.bincount(g_int16).reshape((-1,1))
        gpt = self.gpart
        sizes = np.array([gpt[i+1] - gpt[i] for i in range(len(gpt)-1)]).reshape((-1,1))
        if self.dtype==tf.float32:
            np_type = np.float32
        elif self.dtype==tf.float64:
            np_type = np.float64
        grpmat = csc_matrix((np.ones_like(self.g, dtype=np_type), self.g, np.arange(self.g.shape[0]+1))).tocsr().tocoo()
        print (grpmat.shape)
        sqrt_sizes = np.sqrt(sizes)
        if self.partition is None:
            self.grpmat = coo_to_sparsetensor(grpmat)
            with tf.device(self.devices):
                self.sqrt_sizes   = tf.constant(sqrt_sizes, dtype=self.dtype)
                self.grpidx       = tf.constant(self.g) 
                self.grpidx_2d    = tf.reshape(self.grpidx, (-1,1))
                self.max_norms    = tf.constant(self.lam * sqrt_sizes, dtype=self.dtype)
                self.maxynorm = tf.sqrt(tf.reduce_sum(self.max_norms**2))
        else:
            partition = self.partition
            grp_device_partitioner = partitioners.groupvar_partitioner(partition, gpt)
            dual_partitioner       = partitioners.group_partitioner(gpt)
            self.grp_device_part = grp_device_partitioner(len(gpt)-1, len(self.devices))
            grp_device_part = self.grp_device_part

            self.grpmat = distmat.DistSpMat.from_spmatrix(grpmat, self.devices, partitioner_r=grp_device_partitioner,
                partitioner_c=dual_partitioner)

            self.sqrt_sizes = []
            self.grpidx_2d  = []
            self.max_norms  = []
            
            
            for i,d in enumerate(self.devices):
                with tf.device(d):
                    self.sqrt_sizes.append(tf.constant(sqrt_sizes[grp_device_part[i]:grp_device_part[i+1]], dtype=self.dtype))
                    g_sect = self.g[partition[i]:partition[i+1]]
                    g_sect = g_sect - np.min(g_sect)
                    gidx = tf.constant(g_sect)
                    self.grpidx_2d.append(tf.reshape(gidx, (-1,1)))
                    self.max_norms.append(tf.constant(self.lam*sqrt_sizes[grp_device_part[i]:grp_device_part[i+1]], dtype=self.dtype))
            

            self.sqrt_sizes = distmat.DistMat(self.sqrt_sizes)
            self.grpidx_2d = distmat.DistMat(self.grpidx_2d)
            self.max_norms = distmat.DistMat(self.max_norms)
            self.maxynorm = tf.sqrt((self.max_norms**2).reduce_sum())




    def prox(self, pre_prox, scale):
        with tf.name_scope(self.name):
            with tf.name_scope("prox"):
                if self.partition is None:
                    sumsq = tf.sparse_tensor_dense_matmul(self.grpmat, pre_prox**2)
                    norms = tf.sqrt(sumsq)
                    #max_norms = self.lam*self.sqrt_sizes
                    factors = self.max_norms/(tf.maximum(self.max_norms, norms))
                    factors_elem = tf.gather_nd(factors, self.grpidx_2d)
                    #factors_elem = factors[self.grpidx_shared].reshape((-1,1)) # check this line.
                    return pre_prox * factors_elem
                else:
                    sumsq   = distops.spmatmul(self.grpmat, pre_prox**2)
                    norms   = sumsq.apply(tf.sqrt)
                    factors = self.max_norms.apply_binary(norms, lambda x,y: x/(tf.maximum(x,y)))
                    factors_elem = factors.apply_binary(self.grpidx_2d, tf.gather_nd)
                    return pre_prox * factors_elem

    def eval(self, Dx):
        with tf.name_scope(self.name):
            with tf.name_scope("eval"):
                if self.partition is None:
                    Dx_sumsq = tf.sparse_tensor_dense_matmul(self.grpmat, Dx**2)
                    tf.reshape(Dx_sumsq, ())
                    Dx_norms = tf.sqrt(Dx_sumsq)
                    tf.reshape(Dx_norms, ())
                    product = tf.matmul(tf.transpose(self.sqrt_sizes), Dx_norms)
                    return tf.reshape(self.lam*product,())
                else:
                    Dx_sumsq = distops.spmatmul(self.grpmat, Dx**2)
                    Dx_norms = Dx_sumsq.apply(tf.sqrt)
                    product  = self.sqrt_sizes.apply_binary(Dx_norms, lambda x,y: tf.matmul(tf.transpose(x), y))
                    return tf.reshape(self.lam * tf.add_n(product.tensors), ())


class GroupLasso2(PenaltyFunction):
    # grpidx must be an increasing sequence for this implementation
    # segment_sum is not a GPU operation!
    def __init__(self, lam, g, devices='', partition=None, name=None, dtype=tf.float32):
        assert all([g[i+1]>=g[i] for i in range(len(g)-1)])
        self.lam = lam
        self.g = g.flatten()
        self.gpart = gidx_to_partition(self.g)
        self.dtype=dtype
        self.devices=devices
        self.partition = partition
        super().__init__(name)

    def initialize(self):
        #g_int16 = self.g.astype('int16')
        #sizes = np.bincount(g_int16).reshape((-1,1))
        gpt = self.gpart
        sizes = np.array([gpt[i+1] - gpt[i] for i in range(len(gpt)-1)]).reshape((-1,1))
        sqrt_sizes = np.sqrt(sizes)
        if self.partition is None:
            with tf.device(self.devices):
                self.sqrt_sizes   = tf.constant(sqrt_sizes, dtype=self.dtype)
                self.grpidx       = tf.constant(self.g)
                self.grpidx_2d    = tf.reshape(self.grpidx, (-1,1))
                self.max_norms    = tf.constant(self.lam * sqrt_sizes, dtype=self.dtype)
        else:
            partition       = self.partition
            self.grp_device_part = partitioners.groupvar_partitioner(partition, gpt)(len(gpt)-1, len(self.devices))
            grp_device_part = self.grp_device_part
            self.sqrt_sizes = []
            self.grpidx     = []
            self.grpidx_2d  = []
            self.max_norms  = []

            for i, d in enumerate(self.devices):
                with tf.device(d):

                    self.sqrt_sizes.append(tf.constant(sqrt_sizes[grp_device_part[i]:grp_device_part[i+1]], dtype=self.dtype))
                    g_sect = self.g[partition[i]:partition[i+1]]
                    g_sect = g_sect - np.min(g_sect)
                    gidx = tf.constant(g_sect)
                    self.grpidx.append(gidx)
                    self.grpidx_2d.append(tf.reshape(gidx, (-1,1)))
                    self.max_norms.append(tf.constant(self.lam*sqrt_sizes[grp_device_part[i]:grp_device_part[i+1]], dtype=self.dtype))

            self.sqrt_sizes = distmat.DistMat(self.sqrt_sizes)
            self.grpidx = distmat.DistMat(self.grpidx)
            self.grpidx_2d = distmat.DistMat(self.grpidx_2d)
            self.max_norms = distmat.DistMat(self.max_norms)
            


    def prox(self, pre_prox, scale):
        with tf.name_scope(self.name):
            with tf.name_scope("prox"):
                if self.partition is None:
                    sumsq = tf.segment_sum(pre_prox**2, self.grpidx)
                    #sumsq = tf.sparse_tensor_dense_matmul(self.grpmat, pre_prox**2)
                    norms = tf.sqrt(sumsq)
                    factors = self.max_norms/(tf.maximum(self.max_norms, norms))
                    factors_elem = tf.gather_nd(factors, self.grpidx_2d)
                    #factors_elem = factors[self.grpidx_shared].reshape((-1,1)) # check this line.
                    return pre_prox*factors_elem
                else:
                    
                    sumsq = (pre_prox**2).apply_binary(self.grpidx, tf.segment_sum)
                    norms = sumsq.apply(tf.sqrt)
                    factors = self.max_norms.apply_binary(norms, lambda x, y: x/(tf.maximum(x,y)))
                    factors_elem = factors.apply_binary(self.grpidx_2d, tf.gather_nd)
                    return pre_prox*factors_elem
    def eval(self, Dx):
        with tf.name_scope(self.name):
            with tf.name_scope("eval"):
                if self.partition is None:
                    Dx_sumsq = tf.segment_sum(Dx**2, self.grpidx)
                    #Dx_sumsq = tf.sparse_tensor_dense_matmul(self.grpmat, Dx**2)
                    Dx_norms = tf.sqrt(Dx_sumsq)
                    return tf.reshape(self.lam*tf.matmul(tf.transpose(self.sqrt_sizes), Dx_norms), ())
                else:
                    Dx_sumsq = (Dx**2).apply_binary(self.grpidx, tf.segment_sum) 
                    Dx_norms = Dx_sumsq.apply(tf.sqrt)
                    product = self.sqrt_sizes.apply_binary(Dx_norms, lambda x,y: tf.matmul(tf.transpose(x), y))
                    return tf.reshape(self.lam * tf.add_n(product.tensors), ())
