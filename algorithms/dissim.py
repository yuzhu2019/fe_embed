## Yu Zhu, Rice ECE, yz126@rice.edu
## Reference Paper: Developments in the theory of randomized shortest paths with a comparison of graph node distances

import numpy as np

## compute the cost matrix C from the adjacency matrix A
def adj2cost(A, inf=1e8):
    graph_size = np.size(A, 0)
    C = np.zeros((graph_size, graph_size))
    for i in range(graph_size):
        for j in range(graph_size):
            if A[i,j] > 0:
                C[i,j] = 1/A[i,j]
            elif A[i,j] == 0:
                C[i,j] = inf
    
    return C

## compute the RSP dissimilarity
def RSP(A, C, d, beta):
    graph_size = len(d)
    P_ref = np.matmul(np.diag(1/d), A)
    W = np.multiply(P_ref, np.exp(-beta*C))
    Z = np.linalg.inv(np.eye(graph_size) - W)
    S = np.divide(np.matmul(np.matmul(Z,np.multiply(C,W)),Z),Z)
    C_bar = S - np.matmul(np.ones((graph_size,1)), np.diag(S).reshape(1,-1))
    Delta_RSP = (C_bar + C_bar.T)/2
    
    return Delta_RSP

## compute the FE distance
def FE(A, C, d, beta):
    graph_size = len(d)
    P_ref = np.matmul(np.diag(1/d), A)
    W = np.multiply(P_ref, np.exp(-beta*C))
    Z = np.linalg.inv(np.eye(graph_size) - W)
    Z_h = np.matmul(Z, np.diag(1/np.diag(Z)))
    Phi = -1/beta*np.log(Z_h)
    Delta_FE = (Phi + Phi.T)/2
    
    return Delta_FE

def FE_asy(A, C, d, beta):
    graph_size = len(d)
    P_ref = np.matmul(np.diag(1/d), A)
    W = np.multiply(P_ref, np.exp(-beta*C))
    Z = np.linalg.inv(np.eye(graph_size) - W)
    Z_h = np.matmul(Z, np.diag(1/np.diag(Z)))
    Phi = -1/beta*np.log(Z_h)

    return Phi
