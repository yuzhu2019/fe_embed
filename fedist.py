import numpy as np
import networkx as nx
import graph
from algorithms import dissim
import copy
import time

graph_name = 'BlogCatalog'
nx_G = graph.gen_graph(graph_name)
N = len(nx_G)
A = np.array(nx.to_numpy_matrix(nx_G, weight=None))

inf = 1e8
C = dissim.adj2cost(A, inf)
d = np.sum(A, 0)
beta = 1e-2  # parameter


def log_sum_exp(p_ij, c_ij, phi_jt, beta, thr=7):
    x_j = c_ij + phi_jt
    x_s = np.min(x_j)
    y = beta * (x_j - x_s)
    idx = np.where(y < thr)[0]
    phi_it = x_s - 1/beta * np.log(np.sum(p_ij[idx] * np.exp(-y[idx])))
    return phi_it


node_pairs = [np.delete(np.arange(N), i) for i in range(N)]
succ = [np.argwhere(row > 0)[:, 0].tolist() for row in A]

L = 30  # parameter

P = np.matmul(np.diag(1 / d), A)
Phi = (np.ones((N, N)) - np.eye(N)) * inf

base_file = 'mats_t1/FE_nc/' + graph_name + '_' + str(beta)

t1 = time.time()

for l in range(L):
    print(l)
    Phi0 = copy.deepcopy(Phi)
    for i in range(N):
        succ_i = succ[i]
        for t in node_pairs[i]:
            Phi[i, t] = log_sum_exp(P[i, succ_i], C[i, succ_i], Phi0[succ_i, t], beta)
    if l % 10 == 9:  # parameter
        file = base_file + '_' + str(l) + '.npy'
        np.save(file, Phi)

t2 = time.time()
print(t2 - t1)








