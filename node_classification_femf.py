import numpy as np
import graph
import method
import task
import func
import warnings
warnings.filterwarnings("ignore")

#######################################################

# femf_idx_0 = 8
# femf_idx_1 = 8
# femf_idx_2 = 8
# femf_idx_3 = 9

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
# embedding_methods = ['fe_gmf_neg'] * 6
# args = [[1e-2, 3,  70, True, 300, False],
#         [1e-2, 6,  70, True, 300, False],
#         [1e-2, 9,  70, True, 300, False],
#         [1e-2, 12, 70, True, 300, False],
#         [1e-2, 15, 70, True, 300, False],
#         [1e-2, 18, 70, True, 300, False]]
# embedding_methods = ['fe_gmf_neg']
# args = [[1e-2, 6,  70, True, 300, False]] # for PPI
embedding_methods = ['fe_gmf_neg'] * 6
args = [[1e-4, 6,  70, True, 300, False],
        [1e-3, 6,  70, True, 300, False],
        [1e-2, 6,  70, True, 300, False],
        [1e-1, 6,  70, True, 300, False],
        [1e0,  6,  70, True, 300, False],
        [1e1,  6,  70, True, 300, False]]
rounds = 5
test_ratio_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
file_rs = 'node_classification_rs.npy'  
rs_mat = np.load(file_rs)               
assert len(embedding_methods) == len(args)
n_algs = len(embedding_methods)
n_tr = len(test_ratio_list)

#######################################################

nx_G = graph.gen_graph(graph_name)
file_Y = 'labels/' + graph_name + '_Y.npy'
Y = np.load(file_Y)
micro_f1 = {i:np.zeros((n_tr, rounds)) for i in range(n_algs)}
macro_f1 = {i:np.zeros((n_tr, rounds)) for i in range(n_algs)}
for alg_i in range(n_algs):
    print('alg_i = ', alg_i)
    for round_i in range(rounds):
        print('round_i = ', round_i)
        X = method.embed(nx_G, embedding_methods[alg_i], dim, args[alg_i], graph_name, 'nc') 
        rs = rs_mat[round_i]
        micro, macro = task.node_classification(X, Y, test_ratio_list, rs)
        micro_f1[alg_i][:, round_i] = np.average(micro, axis=1)
        macro_f1[alg_i][:, round_i] = np.average(macro, axis=1)
        
#######################################################

# file_micro_f1 = 'record_v2/nc/micro_f1_' + graph_name + '_' + str(dim) + '_femf_gamma.pkl'
# file_macro_f1 = 'record_v2/nc/macro_f1_' + graph_name + '_' + str(dim) + '_femf_gamma.pkl'
#
# func.save_dict(micro_f1, file_micro_f1)
# func.save_dict(macro_f1, file_macro_f1)
    
#######################################################    
    
micro_f1_avg = np.array([np.mean(micro_f1[i], 1) for i in micro_f1])
macro_f1_avg = np.array([np.mean(macro_f1[i], 1) for i in macro_f1])

print(micro_f1_avg)
print(macro_f1_avg)

# old parameters
# embedding_methods = ['fe_gmf_neg'] * 24
# args = [[1e-4, 6,  90, True, 300, False],
#         [1e-4, 10, 90, True, 300, False],
#         [1e-4, 14, 90, True, 300, False],
#         [1e-4, 18, 90, True, 300, False],
#         [1e-3, 6,  90, True, 300, False],
#         [1e-3, 10, 90, True, 300, False],
#         [1e-3, 14, 90, True, 300, False],
#         [1e-3, 18, 90, True, 300, False],
#         [1e-2, 6,  90, True, 300, False],
#         [1e-2, 10, 90, True, 300, False],
#         [1e-2, 14, 90, True, 300, False],
#         [1e-2, 18, 90, True, 300, False],
#         [1e-1, 6,  90, True, 300, False],
#         [1e-1, 10, 90, True, 300, False],
#         [1e-1, 14, 90, True, 300, False],
#         [1e-1, 18, 90, True, 300, False],
#         [1e0,  6,  90, True, 300, False],
#         [1e0,  10, 90, True, 300, False],
#         [1e0,  14, 90, True, 300, False],
#         [1e0,  18, 90, True, 300, False],
#         [1e1,  6,  90, True, 300, False],
#         [1e1,  10, 90, True, 300, False],
#         [1e1,  14, 90, True, 300, False],
#         [1e1,  18, 90, True, 300, False]]
