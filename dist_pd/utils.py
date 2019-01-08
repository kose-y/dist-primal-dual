import tensorflow as tf
import numpy as np
from scipy.sparse import spmatrix
def coo_to_sparsetensor(spm):
    spt = tf.SparseTensor(indices=np.vstack([spm.row, spm.col]).T, values=spm.data, dense_shape=spm.shape)
    return spt

def gidx_to_partition(gidx):
    assert([gidx[i+1]>=gidx[i] for i in range(len(gidx)-1)])
    m = int(max(gidx))
    return [0]+[np.where(gidx<=i)[0].max()+1 for i in range(m+1)]
def find_nearest(arr, val):
    idx = np.abs(arr-val).argmin()
    return arr[idx]
