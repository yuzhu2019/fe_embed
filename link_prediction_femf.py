import numpy as np
import graph
import method
import task
import func
import warnings
warnings.filterwarnings("ignore")

# idx0 = 3
# idx1 = 3
# idx2 = 3
# idx3 = 2

####################################################

graph_name = 'Cora'
dim = 128
# embedding_methods = ['fe_gmf_neg'] * 30
# args = [[1e-4, 6,  90, True, 300, False],
#         [1e-3, 6,  90, True, 300, False],
#         [1e-2, 6,  90, True, 300, False],
#         [1e-1, 6,  90, True, 300, False],
#         [1e0,  6,  90, True, 300, False],
#         [1e1,  6,  90, True, 300, False],
#         [1e-4, 6,  70, True, 300, False],
#         [1e-3, 6,  70, True, 300, False],
#         [1e-2, 6,  70, True, 300, False],
#         [1e-1, 6,  70, True, 300, False],
#         [1e0,  6,  70, True, 300, False],
#         [1e1,  6,  70, True, 300, False],
#         [1e-4, 6,  50, True, 300, False],
#         [1e-3, 6,  50, True, 300, False],
#         [1e-2, 6,  50, True, 300, False],
#         [1e-1, 6,  50, True, 300, False],
#         [1e0,  6,  50, True, 300, False],
#         [1e1,  6,  50, True, 300, False],
#         [1e-4, 6,  30, True, 300, False],
#         [1e-3, 6,  30, True, 300, False],
#         [1e-2, 6,  30, True, 300, False],
#         [1e-1, 6,  30, True, 300, False],
#         [1e0,  6,  30, True, 300, False],
#         [1e1,  6,  30, True, 300, False],
#         [1e-4, 6,  10, True, 300, False],
#         [1e-3, 6,  10, True, 300, False],
#         [1e-2, 6,  10, True, 300, False],
#         [1e-1, 6,  10, True, 300, False],
#         [1e0,  6,  10, True, 300, False],
#         [1e1,  6,  10, True, 300, False]]
embedding_methods = ['fe_gmf_neg'] * 6
args = [[1e-1, 3,  70, True, 300, False],
        [1e-1, 6,  70, True, 300, False],
        [1e-1, 9,  70, True, 300, False],
        [1e-1, 12, 70, True, 300, False],
        [1e-1, 15, 70, True, 300, False],
        [1e-1, 18, 70, True, 300, False]]
# embedding_methods = ['fe_gmf_neg']
# args = [[1e-1, 6,  70, True, 300, False]]
# embedding_methods = ['fe_gmf_neg'] * 6
# args = [[1e-4, 6,  70, True, 300, False],
#         [1e-3, 6,  70, True, 300, False],
#         [1e-2, 6,  70, True, 300, False],
#         [1e-1, 6,  70, True, 300, False],
#         [1e0,  6,  70, True, 300, False],
#         [1e1,  6,  70, True, 300, False]]
rounds = 10
test_ratio = 0.3
operators = ['average', 'hadamard', 'weighted_l1', 'weighted_l2']
assert len(embedding_methods) == len(args)
n_algs = len(embedding_methods)
n_ops = len(operators)
dir_preprocessed = 'LP_data/' + graph_name + '_' + str(test_ratio) + '/'

####################################################

auc = {i:np.zeros((n_ops, rounds)) for i in range(n_algs)}
for round_i in range(rounds):
    print('round_i = ', round_i)
    # load data
    f_train_graph  = dir_preprocessed + 'train_graph_'  + str(round_i) + '.gpickle'
    f_train_edges  = dir_preprocessed + 'train_edges_'  + str(round_i) + '.txt'
    f_train_labels = dir_preprocessed + 'train_labels_' + str(round_i) + '.txt'
    f_test_edges   = dir_preprocessed + 'test_edges_'   + str(round_i) + '.txt'
    f_test_labels  = dir_preprocessed + 'test_labels_'  + str(round_i) + '.txt'
    train_graph  = func.load_list(f_train_graph)
    train_edges  = func.load_list(f_train_edges)
    train_labels = func.load_list(f_train_labels)
    test_edges   = func.load_list(f_test_edges)
    test_labels  = func.load_list(f_test_labels)
    # return node_id
    node_list = list(train_graph.nodes())
    node_id = {node_list[i]:i for i in range(len(node_list))}
    for alg_i in range(n_algs):
        print('alg_i = ', alg_i)
        node_vectors = method.embed(train_graph, embedding_methods[alg_i], dim, args[alg_i]+[round_i, test_ratio], graph_name, 'lp')
        for op_i in range(n_ops):
            X_train, X_test = task.lp_embed_edge(node_vectors, node_id, train_edges, test_edges, operators[op_i])
            auc[alg_i][op_i, round_i] = task.lp_auc(X_train, train_labels, X_test, test_labels)

####################################################        

file_auc = 'record_v2/lp_' + str(test_ratio) + '/auc_' + str(graph_name) + '_' + str(dim) + '_femf_gamma' + '.pkl'
func.save_dict(auc, file_auc)

#################################################### 

avg_auc = {i:np.mean(auc[i], 1) for i in range(n_algs)}
print(avg_auc)


