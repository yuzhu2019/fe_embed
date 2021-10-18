## The proposed algorithm and baseline algorithms
## Yu Zhu, Rice ECE, yz126@rice.edu
## 2020

import numpy as np
import networkx as nx
import scipy.io
from gensim.models import Word2Vec
from algorithms import dissim, gmf, grarep, netmf, node2vec, gmf2, gmf3
import math
from scipy import sparse
import random


def embed(nx_G, alg, dim, args, graph_name, task):
    '''
    Input
     - nx_G: the networkx graph
     - alg: select the embedding algorithm
     - dim: the embedding dimension
     - args: other parameters
     - graph_name
     - task: node classification (nc), link prediction (lp)
    Output
     - node_vectors: node embeddings
    '''

    graph_size = len(nx_G)
    node_vectors = np.zeros((graph_size, dim))  # numpy.ndarray

    #################################################################################
    #############                         node2vec                      #############
    #################################################################################

    # reference: https://radimrehurek.com/gensim/models/word2vec.html

    if alg == 'node2vec':

        [num_walks, walk_length, p, q, window_size, negative, n_iters, n_workers] = args
        G = node2vec.Graph(nx_G, False, p, q)  # assume the graph is undirected
        G.preprocess_transition_probs()
        walks = G.simulate_walks(num_walks, walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=dim, window=window_size, min_count=0, sg=1, negative=negative, workers=n_workers,
                         iter=n_iters)

        ns = 0
        for node in nx_G.nodes():
            node_vectors[ns, :] = model.wv[str(node)]
            ns += 1

    #################################################################################
    #############                          NetMF                        #############
    #################################################################################

    # T: the context window size
    # b: the number of negative samples in skip-gram
    # h: approximate using top-h eigenpairs

    elif alg == 'NetMF_small':

        [T, b] = args
        A = nx.to_scipy_sparse_matrix(nx_G, weight='weight')
        deepwalk_matrix = netmf.direct_compute_deepwalk_matrix(A, window=T, b=b)
        node_vectors = netmf.svd_deepwalk_matrix(deepwalk_matrix, dim=dim)

    elif alg == 'NetMF_large':

        [T, b, h] = args
        A = nx.to_scipy_sparse_matrix(nx_G, weight='weight')
        vol = float(A.sum())
        evals, D_rt_invU = netmf.approximate_normalized_graph_laplacian(A, rank=h, which="LA")
        deepwalk_matrix = netmf.approximate_deepwalk_matrix(evals, D_rt_invU, window=T, vol=vol, b=b)
        node_vectors = netmf.svd_deepwalk_matrix(deepwalk_matrix, dim=dim)

    #################################################################################
    #############                          GraRep                       #############
    #################################################################################

    # K: maximum transition step
    # _lambda: the number of negative samples

    elif alg == 'GraRep':

        [K, _lambda] = args
        assert dim % K == 0
        A = np.array(nx.to_numpy_matrix(nx_G, weight='weight'))
        node_vectors = grarep.GraRep(A, K, _lambda, int(dim / K))

    #################################################################################
    #############                          HOPE                         #############
    #################################################################################

    elif alg == 'HOPE':

        if task == 'nc':
            [ratio] = args
            file_hope = 'mats/Katz_nc/embeddings/' + graph_name + '_' + str(dim) + '_' + str(ratio) + '.mat'
        elif task == 'lp':
            [ratio, round_i, tr] = args
            file_hope = 'mats/Katz_lp_' + str(tr) + '/embeddings/' + graph_name + '_' + str(dim) + '_' + str(
                ratio) + '_' + str(round_i) + '.mat'
        else:
            print('Error! Check the task name!')

        node_vectors = scipy.io.loadmat(file_hope)['U']

    #################################################################################
    #############                         Proposed                      #############
    #################################################################################

    # elif alg == 'fe_gmf': # the old version
    #
    #     if task == 'nc':
    #         [beta, offdmax, is_symmetric, n_iters, plot] = args
    #         file_fe = 'mats/FE_nc/' + graph_name + '_' + str(beta) + '.npy'
    #     elif task == 'lp':
    #         [beta, offdmax, is_symmetric, n_iters, plot, round_i, tr] = args
    #         file_fe = 'mats/FE_lp_' + str(tr) + '/' + graph_name + '_' + str(beta) + '_' + str(round_i) + '.npy'
    #     else:
    #         print('Error! Check the task name!')
    #
    #     K = np.load(file_fe)
    #     K0 = K - np.diag(np.diag(K))
    #     K_offd_max = np.max(K0)
    #     shift = offdmax / K_offd_max
    #     Nij_p_np = np.exp(K0 * shift)
    #     Nij_n_np = np.ones(np.shape(K))
    #     model = gmf.GMF(Nij_p_np, Nij_n_np, embed_dim=dim, is_symmetric=is_symmetric, n_iters=n_iters, plot=plot)
    #     node_vectors = model.V.numpy()

    elif alg == 'fe_gmf_neg':

        # load the FE distance matrix
        if task == 'nc':
            [beta, offdmax, offset, is_symmetric, n_iters, plot] = args
            file_fe = 'mats_v2/FE_nc/' + graph_name + '_' + str(beta) + '.npy'
        elif task == 'lp':
            [beta, offdmax, offset, is_symmetric, n_iters, plot, round_i, tr] = args
            file_fe = 'mats_v2/FE_lp_' + str(tr) + '/' + graph_name + '_' + str(beta) + '_' + str(round_i) + '.npy'
        else:
            print('Error! Check the task name!')

        Dmat = np.load(file_fe)  # the FE distance matrix - non-negative - the diagonal entries are zero
        num_nodes = np.shape(Dmat)[0]
        iu = np.triu_indices(num_nodes, 1)
        sDmat = -Dmat + np.percentile(Dmat[iu], offset)
        sDmat = sDmat - np.diag(np.diag(sDmat))
        scale = offdmax / np.max(sDmat)
        assert scale > 0
        ssDmat = scale * sDmat
        print(np.max(ssDmat), np.sum(ssDmat>0)/num_nodes/(num_nodes-1)*100)
        Nij_p_np = np.exp(ssDmat)
        Nij_n_np = np.ones(np.shape(Dmat))
        model = gmf.GMF(Nij_p_np, Nij_n_np, embed_dim=dim, is_symmetric=is_symmetric, n_iters=n_iters, plot=plot)
        node_vectors = model.V.numpy()





    # elif alg == 'fe_gmf_neg_test3':
    #
    #     # tau : number of features
    #     if task == 'nc':
    #         [beta, offdmax, offset, n_iters, plot, tau, rs] = args
    #         file_fe = 'mats_v2/FE_nc/' + graph_name + '_' + str(beta) + '.npy'
    #     elif task == 'lp':
    #         [beta, offdmax, offset, n_iters, plot, tau, rs, round_i, tr] = args
    #         file_fe = 'mats_v2/FE_lp_' + str(tr) + '/' + graph_name + '_' + str(beta) + '_' + str(round_i) + '.npy'
    #     else:
    #         print('Error! Check the task name!')
    #
    #     fDmat = np.load(file_fe)  # num_nodes x num_features
    #     random.seed(rs)
    #     ts = random.sample(range(np.shape(fDmat)[0]), tau)
    #     Dmat = fDmat[:, ts]
    #     sDmat = -Dmat + np.percentile(Dmat, offset)
    #     scale = offdmax / np.max(sDmat)
    #     assert scale > 0
    #     ssDmat = scale * sDmat
    #
    #     Nij_p_np = np.exp(ssDmat)
    #     Nij_n_np = np.ones(np.shape(Dmat))
    #     model = gmf3.GMF(Nij_p_np, Nij_n_np, embed_dim=dim, n_iters=n_iters, plot=plot)
    #     node_vectors = model.V.numpy()



    # elif alg == 'dw_gmf':  # use exact DeepWalk matrix
    #
    #     if task == 'nc':
    #         [T, b, shift, is_symmetric, n_iters, plot] = args
    #         file_dw = 'mats/DW_nc/' + graph_name + '_T' + str(T) + '_b' + str(b) + '.npy'
    #     elif task == 'lp':
    #         [T, b, shift, is_symmetric, n_iters, plot, round_i, tr] = args
    #         file_dw = 'mats/DW_lp_' + str(tr) + '/' + graph_name + '_T' + str(T) + '_b' + str(b) + '_' + str(
    #             round_i) + '.npy'
    #     else:
    #         print('Error! Check the task name!')
    #
    #     K = np.load(file_dw)
    #     K0 = K - np.diag(np.diag(K))
    #     Nij_p_np = K0 ** shift
    #     Nij_n_np = np.ones(np.shape(K))
    #     model = gmf.GMF(Nij_p_np, Nij_n_np, embed_dim=dim, is_symmetric=is_symmetric, n_iters=n_iters, plot=plot)
    #     node_vectors = model.V.numpy()

    # elif alg == 'fe_svd':
    #
    #     if task == 'nc':
    #         [beta] = args
    #         file_fe = 'mats/FE_nc/' + graph_name + '_' + str(beta) + '.npy'
    #     elif task == 'lp':
    #         [beta, round_i, tr] = args
    #         file_fe = 'mats/FE_lp_' + str(tr) + '/' + graph_name + '_' + str(beta) + '_' + str(round_i) + '.npy'
    #     else:
    #         print('Error! Check the task name!')
    #
    #     K = np.load(file_fe)
    #     u, s, _ = np.linalg.svd(K)
    #     u = u @ np.diag(s ** 0.5)
    #     node_vectors = u[:, :dim]

    # elif alg == 'fe_ssvd':
    #
    #     if task == 'nc':
    #         [beta] = args
    #         file_fe = 'mats/FE_nc/' + graph_name + '_' + str(beta) + '.npy'
    #     elif task == 'lp':
    #         [beta, round_i, tr] = args
    #         file_fe = 'mats/FE_lp_' + str(tr) + '/' + graph_name + '_' + str(beta) + '_' + str(round_i) + '.npy'
    #     else:
    #         print('Error! Check the task name!')
    #
    #     K = np.load(file_fe)
    #     K[K < 0] = 0
    #     Ks = sparse.csr_matrix(K)
    #     node_vectors = netmf.svd_deepwalk_matrix(Ks, dim=dim)

    else:
        print('Error! Check the method name!')

    return node_vectors
