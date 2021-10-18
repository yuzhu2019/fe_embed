## Use node embeddings for Node Classification and Link Prediction
## Author: Yu Zhu, Rice ECE, yz126@rice.edu
## 2020

import numpy as np
import networkx as nx
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, normalized_mutual_info_score, accuracy_score, roc_auc_score, adjusted_rand_score
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.manifold import TSNE
from munkres import Munkres
import matplotlib.pyplot as plt

##########################################################################
############                   Visualization                   ###########
##########################################################################

def tsne_plot(X, y):
    """
    X: node embeddings
    y: true labels
    """
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    assert len(np.unique(y)) <= len(color_list)
    X_2d = TSNE(n_components=2, random_state=0).fit_transform(X)
    node_colors = [color_list[y_i] for y_i in y]
    plt.figure(figsize=(10,8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=node_colors)
    plt.show()

##########################################################################
############               Graph Reconstruction                ###########
##########################################################################

def argmax_2d(mat, num_largest):
    """ Return the positions of the k maximum values in the 2d array mat. """
    indices = (-mat).argpartition(num_largest, axis=None)[:num_largest]
    x, y = np.unravel_index(indices, mat.shape)
    return x, y

def graph_reconstruction(X, nx_G, list_k, ns, num_runs, rs):
    """
    X: node embeddings
    nx_G: the networkx graph
    list_k: the values of k considered for precision @ k
    ns: num of nodes considered (down-sampling)
    num_runs: num of runs for MC
    rs: random seed
    """
    assert len(X) == len(nx_G)
    num_nodes, num_k = len(X), len(list_k)
    nodes = list(nx_G.nodes())
    random.seed(rs)
    scores = np.zeros((num_k, num_runs))
    for i in range(num_k):
        k = list_k[i]
        for j in range(num_runs):
            sampled_nodes_idx = random.sample(range(num_nodes), ns)
            # true edges -> e_obs
            sampled_nodes = [nodes[idx] for idx in sampled_nodes_idx]
            subg = nx_G.subgraph(sampled_nodes)
            e_obs0 = set(subg.edges())
            e_obs = {tuple(sorted([edge[0], edge[1]])) for edge in e_obs0}
            # print('num of sampled edges', len(e_obs))
            assert k <= len(e_obs)
            # predicted edges -> e_est
            sampled_X = X[sampled_nodes_idx, :]
            S = sampled_X @ sampled_X.T  # the matrix is symmetric
            iu = np.triu_indices(ns)
            S[iu] = np.min(S) - 100  # only consider the lower triangle part
            p1, p2 = argmax_2d(S, k)
            e_est = {tuple(sorted([sampled_nodes[t1], sampled_nodes[t2]])) for t1, t2 in zip(p1, p2)} # check the order  ...
            # compute precision @ k
            scores[i, j] = len(e_est.intersection(e_obs)) / k
    return scores

##########################################################################
############                  Node Clustering                  ###########
##########################################################################

def munkres_assignment(pred, true, verbose=False):
    """
    Assign cluster labels by the Hungarian method. Adapted from https://stackoverflow.com/questions/55258457/find-mapping-that-translates-one-list-of-clusters-to-another-in-python
    :param pred: predicted cluster indices, numpy array, range: 0 - (num of clusters -1)
    :param true: true cluster indices, numpy array, range: 0 - (num of clusters -1)
    :return matched: reordered label predictions
    """
    m = Munkres()
    contmat = contingency_matrix(pred, true)
    mdict = dict(m.compute(contmat.max() - contmat))
    matched = np.sum([(pred == p)*t for p, t in mdict.items()], axis=0)
    if verbose:
        print("matching cluster labels:", ", ".join(["pred %d --> %d"%(p, t) for p,t in mdict.items()]),'\n')
    return matched

def node_clustering(X, y, k, metrics, rs=0, ss=False):
    """
    X: node embeddings, num of nodes x embedding dim
    y: true labels, num of nodes, 1d array
    k: the num of clusters
    metrics: a list of metrics for evaluating clustering accuracy
    rs: the random state in KMeans
    """
    assert k == len(np.unique(y))  # check the number of clusters
    if ss:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        kmeans = KMeans(n_clusters=k, random_state=rs).fit(Xs)
    else:
        kmeans = KMeans(n_clusters=k, random_state=rs).fit(X)
    y_est_raw = kmeans.labels_  # predicted labels
    y_est = munkres_assignment(y_est_raw, y)  # align the labels
    scores = {}
    for metric in metrics:
        if metric == 'NMI':
            scores[metric] = normalized_mutual_info_score(y, y_est)
        elif metric == 'ACC':
            scores[metric] = accuracy_score(y, y_est)
        elif metric == 'ARI':
            scores[metric] = adjusted_rand_score(y, y_est)
        elif metric == 'F1_weighted':
            scores[metric] = f1_score(y, y_est, average='weighted')
        elif metric == 'F1_micro':
            scores[metric] = f1_score(y, y_est, average='micro')
        elif metric == 'F1_macro':
            scores[metric] = f1_score(y, y_est, average='macro')
        else:
            raise NotImplemented
    return scores

##########################################################################
############                 Node Classification               ###########
##########################################################################

def evaluate_nc(X, Y, test_ratio, rs):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=1 - test_ratio, test_size=test_ratio, random_state=rs)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=100))  # default 100

    clf.fit(X_train, Y_train)
    proba = clf.predict_proba(X_test)
    top_k = Y_test.sum(axis=1)
    prediction = np.zeros(np.shape(Y_test), dtype=int)
    for i in range(len(Y_test)):
        for j in proba[i].argsort()[-top_k[i]:]:
            prediction[i, j] = 1

    micro = f1_score(Y_test, prediction, average='micro')
    macro = f1_score(Y_test, prediction, average='macro')

    return micro, macro

def node_classification(X, Y, test_ratio_list, rs):
    """
     - X (np array): node embeddings
     - Y (np array): node labels
     - test_ratio_list (list): the list of different test ratios
     - rs (list): the list of random states (fix the train test split for different algorithms)
    """
    n_tr = len(test_ratio_list)
    rounds = len(rs)
    micro = np.zeros((n_tr, rounds))
    macro = np.zeros((n_tr, rounds))
    for i in range(n_tr):
        test_ratio = test_ratio_list[i]
        for j in range(rounds):
            score = evaluate_nc(X, Y, test_ratio, rs[j])
            micro[i, j] = score[0]
            macro[i, j] = score[1]

    return micro, macro

##########################################################################
############                   Link Prediction                 ###########
##########################################################################    

def lp_split_train_test(nx_G, test_ratio, rs):
    total_edges = list(nx_G.edges())
    train_edges, test_edges = train_test_split(total_edges, train_size=1 - test_ratio, test_size=test_ratio, random_state=rs)

    train_graph = nx.Graph()
    for edge in train_edges:
        edge_weight = nx_G.edges[edge]['weight']
        train_graph.add_edge(edge[0], edge[1], weight=edge_weight)

    # use the largest connected component for embedding
    node_set = list(max(nx.connected_components(train_graph), key=len))
    train_graph_lcc = train_graph.subgraph(node_set)

    train_edges_pos = list(train_graph_lcc.edges())
    test_edges_pos = [edge for edge in test_edges if edge[0] in node_set and edge[1] in node_set]

    subg = nx_G.subgraph(node_set)
    total_edges_neg = list(nx.non_edges(subg))

    n_train_neg = len(train_edges_pos)
    n_test_neg = len(test_edges_pos)
    edges_neg = random.sample(total_edges_neg, n_train_neg + n_test_neg)
    train_edges_neg = edges_neg[:n_train_neg]
    test_edges_neg = edges_neg[-n_test_neg:]

    train_edges = train_edges_pos + train_edges_neg
    test_edges = test_edges_pos + test_edges_neg
    train_labels = [1] * n_train_neg + [0] * n_train_neg
    test_labels = [1] * n_test_neg + [0] * n_test_neg

    train_edges, train_labels = shuffle(train_edges, train_labels)
    test_edges, test_labels = shuffle(test_edges, test_labels)

    return train_graph_lcc, train_edges, train_labels, test_edges, test_labels


def embed_edge(nv1, nv2, operator='hadamard'):
    if operator == 'average':
        edge_vector = (nv1 + nv2) / 2
    elif operator == 'hadamard':
        edge_vector = np.multiply(nv1, nv2)
    elif operator == 'weighted_l1':
        edge_vector = np.absolute(nv1 - nv2)
    elif operator == 'weighted_l2':
        edge_vector = (nv1 - nv2) ** 2

    return edge_vector


def lp_embed_edge(node_vectors, node_id, train_edges, test_edges, operator):
    dim = np.size(node_vectors, 1)

    n_train = len(train_edges)
    X_train = np.zeros((n_train, dim))
    for i in range(n_train):
        nv1 = node_vectors[node_id[train_edges[i][0]], :]
        nv2 = node_vectors[node_id[train_edges[i][1]], :]
        X_train[i, :] = embed_edge(nv1, nv2, operator)

    n_test = len(test_edges)
    X_test = np.zeros((n_test, dim))
    for i in range(n_test):
        nv1 = node_vectors[node_id[test_edges[i][0]], :]
        nv2 = node_vectors[node_id[test_edges[i][1]], :]
        X_test[i, :] = embed_edge(nv1, nv2, operator)

    return X_train, X_test


def lp_auc(X_train, Y_train, X_test, Y_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train, Y_train)
    proba = clf.predict_proba(X_test)
    auc = roc_auc_score(Y_test, proba[:, 1])

    return auc
