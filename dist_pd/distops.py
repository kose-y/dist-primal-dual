import tensorflow as tf
import dist_pd.distmat as distmat
from collections import defaultdict

def matmul(A, B, transpose_A=False, transpose_B=False, master='/gpu:0'):
    """
    distributed matrix multiplication.
    A: DistMat, 
    B: single tensor or a list of tensors.
    Note: returns a single tensor or a list of tensors, Not a DistMat.
    """
    if isinstance(A, tf.Tensor) or isinstance(A, tf.Variable):
        if isinstance(B, tf.Tensor) or isinstance(B, tf.Variable):
            return tf.matmul(A, B)
        else: 
            raise NotImplementedError
    if transpose_B: 
        raise NotImplementedError
    else:
        if transpose_A: # distributed dim is inner axis
            if isinstance(B, tf.Tensor) or isinstance(B, tf.Variable):
                # broadcast
                partial_sums = []
                for i, t in enumerate(A.tensors):
                    with tf.device(t.device):
                        partial_sums.append(tf.matmul(t, B[A.partition[i]:A.partition[i+1],:], transpose_a=True))
                with tf.device(master):
                    return tf.add_n(partial_sums) 
            else:
                partial_sums = []
                for t_A, t_B in zip(A.tensors, B.tensors):
                    #print(t_A.device)
                    #print(t_B.device)
                    #assert t_A.device == t_B.device
                    with tf.device(t_A.device):
                        partial_sums.append(tf.matmul(t_A, t_B, transpose_a=True))
                with tf.device(master):
                    return tf.add_n(partial_sums)
                # distributed computation necessary
                #return tf.add_n([tf.matmul(Apart, Bpart) for Apart, Bpart in zip(A.tensors, B.tensors)])
        else: # non-distributed dim is inner axis. merely broacast B. 
            if isinstance(B, tf.Tensor) or isinstance(B, tf.Variable):
                slices = []
                for t in A.tensors:
                    with tf.device(t.device):
                        slices.append(tf.matmul(t, B))
                return distmat.DistMat(slices)
            else: 
                raise NotImplementedError

def spmatmul(D, x, transpose_A=False, transpose_B=False):
    """
    distributed sparse matrix times a dense vector.
    Note: The behavior of this operation is based on an undocumented feature of `scatter_nd`. If something changes in the
    implementation of `scatter_nd`, we should change this implementation. 
    """
    #print(type(D))
    assert isinstance(D, distmat.DistSpMat)
    if transpose_B:
        raise NotImplementedError
    if isinstance(x, distmat.DistMat):
        # TODO: check validity
        # take the list of tensors
        x = x.tensors
        #print([t.shape for t in x])
    if isinstance(x, list):
        pass
    else:
        x = [x]

    Dxparts = defaultdict(list)
    outlist = []
    #print(type(x))
    xcols = x[0].shape[1]
    if isinstance(xcols, tf.Dimension):
        xcols = xcols.value

    if transpose_A:
        # piecewise computation
        for i in range(len(D.devices_r)):
            xpiece = x[i]
            for j in range(len(D.devices_c)):
                if D.D_tensors[i][j]:
                    Dt_block = D.Dt_tensors[i][j]
                    
                    with tf.device(D.devices_r[i]):
                        Dxparts[j].append(tf.sparse_tensor_dense_matmul(Dt_block, xpiece))
        # scatter
        for j in range(len(D.devices_c)):
            with tf.device(D.devices_c[j]):
                Dxdata = tf.concat(Dxparts[j], 0)
                Dxidx  = tf.reshape(D.Dt_nz_r_all[j], (-1,1))
                rows  = D.partition_c[j+1] - D.partition_c[j]
                outlist.append(tf.scatter_nd(Dxidx, Dxdata, [rows, xcols]))
        
    else:
        # piecewise computation
        for j in range(len(D.devices_c)):
            xpiece = x[j]
            for i in range(len(D.devices_r)):
                if D.D_tensors[i][j]:
                    D_block = D.D_tensors[i][j]
                    with tf.device(D.devices_c[j]):
                        Dxparts[i].append(tf.sparse_tensor_dense_matmul(D_block, xpiece))
        # scatter
        for i in range(len(D.devices_r)):
            with tf.device(D.devices_r[i]):
                Dxdata = tf.concat(Dxparts[i], 0)
                Dxidx  = tf.reshape(D.D_nz_r_all[i], (-1,1))
                rows = D.partition_r[i+1]-D.partition_r[i]
                outlist.append(tf.scatter_nd(Dxidx, Dxdata, [rows, xcols]))
    return distmat.DistMat(outlist)

def matmul_dropout(A, B, rate=1.0, noise_shape=None, transpose_A=False, transpose_B=False, master='/gpu:0'):
    """
    distributed matrix multiplication.
    A: DistMat, 
    B: single tensor or a list of tensors.
    rate: keep prob.
    noise_shape : 'row' or 'None'. implementation for 'col' is incomplete. 
    Note: returns a single tensor or a list of tensors, Not a DistMat.
    """
    noise_shape_slice=None
    if isinstance(A, tf.Tensor) or isinstance(A, tf.Variable):
        if isinstance(B, tf.Tensor) or isinstance(B, tf.Variable):
            if noise_shape =='row':
                noise_shape_slice = [A.shape[0], 1]
            drop_A = tf.nn.dropout(A, rate, noise_shape_slice)
            return tf.matmul(A, B)
        else: 
            raise NotImplementedError
    if transpose_B: 
        raise NotImplementedError
    else:
        if transpose_A: # distributed dim is inner axis
            if isinstance(B, tf.Tensor) or isinstance(B, tf.Variable):
                # broadcast
                partial_sums = []
                for i, t in enumerate(A.tensors):
                    with tf.device(t.device):
                        if noise_shape == 'row':
                            noise_shape_slice = [t.shape[0].value, 1]
                        
                        t_drop = tf.nn.dropout(t, rate, noise_shape_slice)
                        partial_sums.append(tf.matmul(t_drop, B[A.partition[i]:A.partition[i+1],:], transpose_a=True))
                with tf.device(master):
                    return tf.add_n(partial_sums) 
            else:
                partial_sums = []
                for t_A, t_B in zip(A.tensors, B.tensors):
                    #print(t_A.device)
                    #print(t_B.device)
                    #assert t_A.device == t_B.device
                    with tf.device(t_A.device):
                        if noise_shape == 'row':
                            noise_shape_slice = [t_A.shape[0].value, 1]
                        t_A_drop = tf.nn.dropout(t_A, rate, noise_shape_slice)
                        partial_sums.append(tf.matmul(t_A_drop, t_B, transpose_a=True))
                with tf.device(master):
                    return tf.add_n(partial_sums)
                # distributed computation necessary
                #return tf.add_n([tf.matmul(Apart, Bpart) for Apart, Bpart in zip(A.tensors, B.tensors)])
        else: # non-distributed dim is inner axis. merely broacast B. 
            if isinstance(B, tf.Tensor) or isinstance(B, tf.Variable):
                slices = []
                for t in A.tensors:
                    with tf.device(t.device):
                        if noise_shape == 'row':
                            noise_shape_slice = [t.shape[0].value, 1]
                        t_drop = tf.nn.dropout(t, rate, noise_shape_slice)
                        slices.append(tf.matmul(t_drop, B))
                return distmat.DistMat(slices)
            else: 
                raise NotImplementedError


zero = tf.constant(0, dtype=tf.int64)

def spmatmul_dropout(D, x, rate=1.0, transpose_A=False, transpose_B=False):
    """
    distributed sparse matrix times a dense vector.
    Note: The behavior of this operation is based on an undocumented feature of `scatter_nd`. If something changes in the
    implementation of `scatter_nd`, we should change this implementation.
    rate: proportion to keep! 
    """
    #print(type(D))
    assert isinstance(D, distmat.DistSpMat)
    if transpose_B:
        raise NotImplementedError
    if isinstance(x, distmat.DistMat):
        # TODO: check validity
        # take the list of tensors
        x = x.tensors
        #print([t.shape for t in x])
    if isinstance(x, list):
        pass
    else:
        x = [x]

    Dxparts = defaultdict(list)
    outlist = []
    #print(type(x))
    xcols = x[0].shape[1]
    if isinstance(xcols, tf.Dimension):
        xcols = xcols.value

    if transpose_A:
        # piecewise computation
        for i in range(len(D.devices_r)):
            xpiece = x[i]
            for j in range(len(D.devices_c)):
                if D.D_tensors[i][j]:
                    Dt_block = D.Dt_tensors[i][j]
                    nonzero_elems = Dt_block.values.shape[0].value
                    with tf.device('/cpu:0'):
                        select = tf.not_equal(tf.multinomial(tf.log([[1-rate, rate]]), nonzero_elems)[0], zero)
                        Dt_block_drop = tf.sparse_retain(Dt_block, select)
                    
                    
                    with tf.device(D.devices_r[i]):
                        Dxparts[j].append(tf.sparse_tensor_dense_matmul(Dt_block_drop, xpiece))
        # scatter
        for j in range(len(D.devices_c)):
            with tf.device(D.devices_c[j]):
                Dxdata = tf.concat(Dxparts[j], 0)
                Dxidx  = tf.reshape(D.Dt_nz_r_all[j], (-1,1))
                rows  = D.partition_c[j+1] - D.partition_c[j]
                outlist.append(tf.scatter_nd(Dxidx, Dxdata, [rows, xcols]))
        
    else:
        # piecewise computation
        for j in range(len(D.devices_c)):
            xpiece = x[j]
            for i in range(len(D.devices_r)):
                if D.D_tensors[i][j]:
                    D_block = D.D_tensors[i][j]
                    nonzero_elems = D_block.values.shape[0].value
                    with tf.device('/cpu:0'):
                        select = tf.not_equal(tf.multinomial(tf.log([[1-rate, rate]]), nonzero_elems)[0], zero)
                        D_block_drop = tf.sparse_retain(D_block, select)
                    with tf.device(D.devices_c[j]):
                        Dxparts[i].append(tf.sparse_tensor_dense_matmul(D_block_drop, xpiece))
        # scatter
        for i in range(len(D.devices_r)):
            with tf.device(D.devices_r[i]):
                Dxdata = tf.concat(Dxparts[i], 0)
                Dxidx  = tf.reshape(D.D_nz_r_all[i], (-1,1))
                rows = D.partition_r[i+1]-D.partition_r[i]
                outlist.append(tf.scatter_nd(Dxidx, Dxdata, [rows, xcols]))
    return distmat.DistMat(outlist)/rate # scale by rate!


                    

                    

