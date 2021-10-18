## Algorithm: GraRep
## Original code written in Matlab: https://github.com/ShelsonCao/GraRep
## Paper: GraRep: Learning Graph Representations with Global Structural Information
## Yu Zhu, Rice ECE, yz126@rice.edu

import numpy as np
import copy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

eps = 1e-10 # smaller than this value -> zero

def sparse_svd(X, d, set0):
    '''
    X: the matrix to be decomposed (numpy array)
    d: consider the top d singular values
    set0: whether set negative values to zero
    '''
    Xs = copy.deepcopy(X)
    if set0:
        Xs[Xs < 0] = 0                # set negative entries to zero
    Xs = csc_matrix(Xs, dtype=float)  # convert to sparse matrix
    u, s, vt = svds(Xs, k=d)          
    u = u[:,::-1]            
    s = s[::-1]                       # sort in decreasing order (not necessary)
    W = np.matmul(u, np.diag(np.sqrt(s)))
    return W

def GraRep(S, K, _lambda, d):
    '''
    S: adjacency matrix (numpy array)
    K: maximum transition step 
    _lambda: the number of negative samples
    d: embedding dimension 
    '''
    N = np.size(S, 0)  # the graph size
    beta = _lambda/N   # log shifted factor

    D = np.sum(S, 1) 
    A = np.matmul(np.diag(1./D), S) 
    
    A_k = np.zeros((K, N, N))
    A_k[0,:,:] = copy.deepcopy(A)
    for k in range(1, K):
        A_k[k,:,:] = np.matmul(A_k[k-1,:,:], A) 
        
    W = np.zeros((N, K*d))
    for k in range(K):
        tau_k = np.sum(A_k[k,:,:], 0) 
        X_k = np.log(A_k[k,:,:]/tau_k) - np.log(beta)
        W_k = sparse_svd(X_k, d, True)
        W_k_norm = np.linalg.norm(W_k, axis=1)
        for n in range(N):
            if W_k_norm[n] > eps:
                W[n, k*d:(k+1)*d] = W_k[n]/W_k_norm[n]
            else:
                W[n, k*d:(k+1)*d] = np.ones(d) 
        
    return W




