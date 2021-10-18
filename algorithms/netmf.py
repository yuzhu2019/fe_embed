## Algorithm: NetMF
## The original code is downloaded from https://github.com/xptree/NetMF
## Paper: Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec

import numpy as np
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import theano
from theano import tensor as T
theano.config.exception_verbosity='high'

##############################################################################
###########    Exact DeepWalk (NetMF for a Small Window Size T)    ###########
##############################################################################

def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True) 
    X = sparse.identity(n) - L # D^(-1/2) @ A @ D^(-1/2)
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        X_power = X_power.dot(X)
        S += X_power        
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T) 
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y) 

##############################################################################
#########   Approximate DeepWalk (NetMF for a Large Window Size T)   #########
##############################################################################

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L 
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    D_rt_inv = sparse.diags(d_rt ** -1) 
    D_rt_invU = D_rt_inv.dot(evecs)     
    return evals, D_rt_invU

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    return evals

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)    
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b) 
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    return sparse.csr_matrix(Y)

##############################################################################
#####################              Sparse SVD            #####################
##############################################################################

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    return sparse.diags(np.sqrt(s)).dot(u.T).T


## new added by Yu

def comp_S_DW(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True) 
    X = sparse.identity(n) - L # D^(-1/2) @ A @ D^(-1/2)
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        X_power = X_power.dot(X)
        S += X_power        
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T) 
    return M.todense()


