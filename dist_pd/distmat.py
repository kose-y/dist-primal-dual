import numpy as np
import tensorflow as tf
import scipy
from collections import defaultdict
from dist_pd.utils import coo_to_sparsetensor
import dist_pd.partitioners as partitioners

class DistMat:
    """
    Distributed Dense Matrix.
    Only consider 1d partitioning. 
    """
    def __init__(self, list_of_tensors, sess=None):
        #assert all([len(t.shape)==2 or t.shape.dims[0]==tf.Dimension(2) for t in list_of_tensors])
        self.tensors = list_of_tensors
        self.sess = sess
        if any([len(t.shape)==2 or (len(self.tensors[0].shape.dims)>0 and t.shape.dims[0]==tf.Dimension(2)) for t in list_of_tensors]):
            assert all([list_of_tensors[0].shape[1] == t.shape[1] for t in list_of_tensors])
        #print([t.shape for t in self.tensors])
        if len(self.tensors[0].shape)<1:
            self.shape=tuple()
        else:
            if self.tensors[0].shape[0].value is None:
                self.shape=None
            else: 
                self.shape = sum([t.shape[0].value for t in self.tensors])
            if (len(self.tensors[0].shape)==2) or (len(self.tensors[0].shape.dims)>0 and self.tensors[0].shape.dims[0]==tf.Dimension(2)) :
                self.shape = self.shape, self.tensors[0].shape[1].value
            else:
                self.shape = (self.shape,)
        if self.shape and self.shape[0] is not None:
            self.part_sizes = [t.shape[0].value for t in self.tensors]
            self.partition = list(np.cumsum([0]+self.part_sizes))
        else:
            self.part_sizes = None
            self.partition = None
    def apply(self, op, *args):
        r = []
        for t in self.tensors:
            with tf.device(t.device):
                r.append(op(t,*args))
        return DistMat(r)
    def apply_binary(self, other, op, *args):
        r = []
        for t1, t2 in zip(self.tensors, other.tensors):
            with tf.device(t1.device):
                r.append(op(t1, t2, *args))
        return DistMat(r)
    def dropout(self, rate, noise_shape_slice=None):
        return self.apply(tf.nn.dropout, rate, noise_shape_slice)
        
    def reduce_sum(self):
        r = []
        for t in self.tensors:
            with tf.device(t.device):
                r.append(tf.reduce_sum(t))
        return tf.reduce_sum(r)
    def _generic_binary(self, other, op):
        if isinstance(other, DistMat):
            return self.apply_binary(other, op)
        else:
            return self.apply(lambda x: op(x, other))
    def __add__(self, other):
        return self._generic_binary(other, lambda x,y: x+y)
    def __mul__(self, other):
        return self._generic_binary(other, lambda x,y: x*y)
    def __sub__(self, other):
        return self._generic_binary(other, lambda x,y: x-y)
    def __truediv__(self, other):
        return self._generic_binary(other, lambda x,y: x/y)
    def __pow__(self, other):
        return self._generic_binary(other, lambda x,y: x**y)
    def __radd__(self, other):
        return self._generic_binary(other, lambda x,y: x+y)
    def __rmul__(self, other):
        return self._generic_binary(other, lambda x,y: x*y)
    def __rsub__(self, other):
        return self._generic_binary(other, lambda x,y: y-x)
    def __rtruediv__(self, other):
        return self._generic_binary(other, lambda x,y: y/x)




    @classmethod    
    def from_dataset(self, dat, axis=0, devices=['/gpu:0'], sess=None, dtype=tf.float32,
                        partitioner=partitioners.default_partitioner):
            
        if sess is None:
            sess = tf.Session()
        #self.sess=sess
        if axis != 0:
            raise NotImplementedError
        shape = dat.shape
        d     = len(devices)
        n     = shape[axis]
        partition = partitioner(n,d)
        #placeholders = []
        variables    = []
        ph_dict = {}

        for i, d in enumerate(devices):
            #print(i)
            with tf.device(d):
                part_rows = partition[i+1]-partition[i]
                if part_rows in ph_dict.keys():
                    ph = ph_dict[part_rows]
                else:
                    ph = tf.placeholder(dtype, shape=(partition[i+1]-partition[i],shape[1] ))
                    ph_dict[part_rows] = ph
                    

                #ph = tf.placeholder(dtype, shape=(partition[i+1]-partition[i],shape[1] ))
                v  = tf.Variable(ph)

                #placeholders.append(ph)
            feed_dict = dict()
            feed_dict[ph] = dat[partition[i]:partition[i+1], :]

            sess.run(v.initializer, feed_dict=feed_dict)
            variables.append(v)

        #initializers = [v.initializer for v in variables]


        #feed_dict = dict()
        #for i, ph in enumerate(placeholders):
            #feed_dict[ph] = dat[partition[i]:partition[i+1], :]

        #sess.run(initializers, feed_dict=feed_dict)
        return DistMat(variables, sess)
    def eval(self,session):
        return [t.eval(session=session) for t in self.tensors]




def csr_nonzero_rows(D):
    assert isinstance(D, scipy.sparse.csr_matrix)
    return np.arange(D.shape[0])[D.indptr[1:]-D.indptr[:-1]>0].astype('int64')


DD = lambda : defaultdict(defaultdict)
DL = lambda : defaultdict(list)
DA = lambda : defaultdict(lambda: np.array(None))
DN = lambda : defaultdict(lambda: None)
DT = lambda : defaultdict(tf.constant([]))

class DistSpMat:
    """
    Distributed Sparse Matrix.
    Assume 2d partitioning.
    """
    def __init__(self, D_tensors, Dt_tensors, D_nonzero_rows, Dt_nonzero_rows, partition_r, partition_c, devices_r, devices_c):
        self.D_tensors = D_tensors
        self.Dt_tensors=Dt_tensors
        self.D_nz_r = D_nonzero_rows
        self.Dt_nz_r = Dt_nonzero_rows
        self.partition_r = partition_r
        self.partition_c = partition_c
        self.devices_r = devices_r
        self.devices_c = devices_c
        self.shape = (self.partition_r[-1], self.partition_c[-1])
        self.D_nz_r_list = defaultdict(list)
        self.Dt_nz_r_list = defaultdict(list)
        
        for i in range(len(self.devices_r)):
            for j in range(len(self.devices_c)):
                if self.D_nz_r[i][j] is not None:
                    self.D_nz_r_list[i].append(self.D_nz_r[i][j])
                    self.Dt_nz_r_list[j].append(self.Dt_nz_r[i][j])
 
        self.D_nz_r_all = defaultdict(DT)
        self.Dt_nz_r_all = defaultdict(DT)
        for i in range(len(self.devices_r)):
            self.D_nz_r_all[i] = tf.concat(self.D_nz_r_list[i],0)
        for j in range(len(self.devices_c)):
            self.Dt_nz_r_all[j] = tf.concat(self.Dt_nz_r_list[j],0)


        
    @classmethod
    def from_spmatrix(self, D, devices_r=['/gpu:0'], devices_c=None, idx_as_tensors=True, 
        partitioner_r=partitioners.default_partitioner, partitioner_c=partitioners.default_partitioner):
        if devices_c is None:
            devices_c = devices_r
        assert isinstance(D, scipy.sparse.spmatrix)
        D_csr  = D.tocsr()
        Dt_csr = D.T.tocsr()
        shape  = D.shape
        nd_r      = len(devices_r)
        nd_c      = len(devices_c)
        n_r     = shape[0]
        n_c     = shape[1]

        partition_r = partitioner_r(n_r, nd_r)
        partition_c = partitioner_c(n_c, nd_c)
        #q_r     = n_r//nd_r
        #q_c     = n_c//nd_c
        #r_r     = n_r%nd_r
        #r_c     = n_c%nd_c
        #n_ris   = [q_r + (1 if i<r_r else 0) for i in range(nd_r)]
        #n_cis   = [q_c + (1 if i<r_c else 0) for i in range(nd_c)]
        #partition_r = np.cumsum([0]+n_ris)
        #partition_c = np.cumsum([0]+n_cis)
        #if grppart: 
        #    for i in range(len(partition_r)):
        #        partition_r[i] = utils.find_nearest(grppart, partition_r[i])
        variables_r    = []
        variables_c    = []
        D_nz_r = csr_nonzero_rows(D_csr)
        D_nz_c = csr_nonzero_rows(Dt_csr)

        # D_blocks: 2d blocks
        # D_nz_r[i][j]: nonzero rows in block (i,j)
        # D_nz_c[i][j]: nonzero cols in block (i,j)
        D_nz_r   = defaultdict(DL)
        D_nz_c   = defaultdict(DL)

        D_tensors = defaultdict(DN)
        Dt_tensors = defaultdict(DN)

        D_nz_r_tensors = defaultdict(DN)
        D_nz_c_tensors = defaultdict(DN)


        for i in range(len(devices_r)):
            for j in range(len(devices_c)):
                block = D_csr[partition_r[i]:partition_r[i+1], partition_c[j]:partition_c[j+1]]
                if len(block.data)!=0:
                    D_block = block
                    Dt_block = block.T.tocsr()
                    
                    D_nz_r[i][j] = csr_nonzero_rows(D_block)
                    D_nz_c[i][j] = csr_nonzero_rows(Dt_block)

                    with tf.device(devices_r[i]):
                        D_nz_r_tensors[i][j] = tf.constant(D_nz_r[i][j])
                    with tf.device(devices_c[j]):
                        D_nz_c_tensors[i][j] = tf.constant(D_nz_c[i][j])

                    D_blocks_short  = D_block[D_nz_r[i][j]]
                    Dt_blocks_short = Dt_block[D_nz_c[i][j]]
                    # the order is already sorted by csr conversion. 
                    with tf.device(devices_r[i]):
                        Dt_tensors [i][j] = coo_to_sparsetensor(Dt_blocks_short.tocoo())
                    with tf.device(devices_c[j]):
                        D_tensors[i][j] = coo_to_sparsetensor(D_blocks_short.tocoo())
        if idx_as_tensors:
            return DistSpMat(D_tensors, Dt_tensors, D_nz_r_tensors, D_nz_c_tensors, partition_r, partition_c, devices_r, devices_c)
        else:
            return DistSpMat(D_tensors, Dt_tensors, D_nz_r, D_nz_c, partition_r, partition_c, devices_r, devices_c)




                

        '''

        for i,d in enumerate(devices_r):
            target = D_csr[partition_r[i]:partition_r[i+1], :]
            target_nzrows = csr_nonzero_rows(target)
            target_nzrowmat = target[target_nz,:]

            with tf.device(d):
                pass
        '''
                
