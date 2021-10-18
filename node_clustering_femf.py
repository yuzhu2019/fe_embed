import numpy as np
import random
import graph
import method
import task
import func

## parameters
graph_name = 'Cora'
dim = 8
# algs = ['fe_gmf_neg'] * 30
# is_symmetric = True
# args = [[1e-4, 6,  90, is_symmetric, 300, False],
#         [1e-3, 6,  90, is_symmetric, 300, False],
#         [1e-2, 6,  90, is_symmetric, 300, False],
#         [1e-1, 6,  90, is_symmetric, 300, False],
#         [1e0,  6,  90, is_symmetric, 300, False],
#         [1e1,  6,  90, is_symmetric, 300, False],
#         [1e-4, 6,  70, is_symmetric, 300, False],
#         [1e-3, 6,  70, is_symmetric, 300, False],
#         [1e-2, 6,  70, is_symmetric, 300, False],
#         [1e-1, 6,  70, is_symmetric, 300, False],
#         [1e0,  6,  70, is_symmetric, 300, False],
#         [1e1,  6,  70, is_symmetric, 300, False],
#         [1e-4, 6,  50, is_symmetric, 300, False],
#         [1e-3, 6,  50, is_symmetric, 300, False],
#         [1e-2, 6,  50, is_symmetric, 300, False],
#         [1e-1, 6,  50, is_symmetric, 300, False],
#         [1e0,  6,  50, is_symmetric, 300, False],
#         [1e1,  6,  50, is_symmetric, 300, False],
#         [1e-4, 6,  30, is_symmetric, 300, False],
#         [1e-3, 6,  30, is_symmetric, 300, False],
#         [1e-2, 6,  30, is_symmetric, 300, False],
#         [1e-1, 6,  30, is_symmetric, 300, False],
#         [1e0,  6,  30, is_symmetric, 300, False],
#         [1e1,  6,  30, is_symmetric, 300, False],
#         [1e-4, 6,  10, is_symmetric, 300, False],
#         [1e-3, 6,  10, is_symmetric, 300, False],
#         [1e-2, 6,  10, is_symmetric, 300, False],
#         [1e-1, 6,  10, is_symmetric, 300, False],
#         [1e0,  6,  10, is_symmetric, 300, False],
#         [1e1,  6,  10, is_symmetric, 300, False]]
# algs = ['fe_gmf_neg'] * 6
# args = [[1e-4, 6,  70, True, 300, False],
#         [1e-3, 6,  70, True, 300, False],
#         [1e-2, 6,  70, True, 300, False],
#         [1e-1, 6,  70, True, 300, False],
#         [1e0,  6,  70, True, 300, False],
#         [1e1,  6,  70, True, 300, False]]
algs = ['fe_gmf_neg'] * 6
args = [[1e-1, 3,  70, True, 300, False],
        [1e-1, 6,  70, True, 300, False],
        [1e-1, 9,  70, True, 300, False],
        [1e-1, 12, 70, True, 300, False],
        [1e-1, 15, 70, True, 300, False],
        [1e-1, 18, 70, True, 300, False]]
assert len(algs) == len(args)
n_algs = len(algs)
rounds = 5

## clustering parameters
if graph_name == 'Citeseer':
    k = 6
elif graph_name == 'Cora':
    k = 7
else:
    raise NotImplemented
metrics = ['NMI', 'ACC', 'ARI', 'F1_weighted']
KMeans_runs = 10
random.seed(0)
rs = [random.sample(range(1000), KMeans_runs) for round_i in range(rounds)]

## load true labels
file_y = 'labels/' + graph_name + '_TrueLabels.npy'
y = np.load(file_y)

## main
nx_G = graph.gen_graph(graph_name)
acc = {alg_i: {metric: np.zeros((rounds, KMeans_runs)) for metric in metrics} for alg_i in range(n_algs)}
for alg_i in range(n_algs):
    print(alg_i)
    for round_i in range(rounds):
        print('round_i = ', round_i)
        X = method.embed(nx_G, algs[alg_i], dim, args[alg_i], graph_name, 'nc')
        for run_i in range(KMeans_runs):
            scores = task.node_clustering(X, y, k, metrics, rs[round_i][run_i], False)  # return a dict
            for metric in metrics:
                acc[alg_i][metric][round_i, run_i] = scores[metric]

## output
acc_avg = {alg_i: {metric: np.mean(acc[alg_i][metric]) for metric in metrics} for alg_i in range(n_algs)}

for alg_i in range(n_algs):
    print(acc_avg[alg_i])

## save data
# if is_symmetric:
#     file_acc = 'record_v2/clustering/' + graph_name + str(dim) + '/femf' + '.pkl'
# else:
#     file_acc = 'record_v2/clustering/' + graph_name + str(dim) + '/femf_asy' + '.pkl'
file_acc = 'record_v2/clustering/' + graph_name + str(dim) + '_femf_gamma_-1' + '.pkl'
func.save_dict(acc, file_acc)
