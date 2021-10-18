## Free Energy Node Embedding via Generalized Skip-gram with Negative Sampling

The code for the paper "Free Energy Node Embedding via Generalized Skip-gram with Negative Sampling" by Yu Zhu, Ananthram Swami, and Santiago Segarra. 

Paper link: https://arxiv.org/pdf/2105.09182.pdf

#### Datasets

The datasets are included in the subfile 'data', which can be read in using 'graph.py'. The dataset information can be obtained by running 'graph_info.ipynb'.

The datasets used in this paper are downloaded from the following links:

CiteSeer, Cora, BlogCatalog: http://zhang18f.myweb.cs.uwindsor.ca/datasets/

PPI: https://snap.stanford.edu/node2vec/

#### Methods

The codes for the proposed method and the baseline methods are included in the subfile 'algorithms'. These methods can be easily called using 'method.py'. The proposed one corresponds to 'fe_gmf_neg'.

The codes for baseline methods are downloaded from the following links:

node2vec: https://github.com/aditya-grover/node2vec

GraRep: https://github.com/ShelsonCao/GraRep

HOPE: https://github.com/ZW-ZHANG/HOPE

NetMF: https://github.com/xptree/NetMF

We implement DeepWalk using the code of node2vec by setting both of the in-out and return parameters to 1.

#### Downstream tasks

The code for downstream tasks is given in 'task.py' including node clustering, node classification, link prediction, etc.

#### A simple example

The code for generating Figure 2 in the manuscript is given in ‘example.ipynb’.

#### Performance evaluation

The codes (main function) for the performance evaluation of the proposed method are given in 'node_clustering_femf.py', 'node_classification_femf.py' and 'link_prediction_femf.py'. Some preparatory work is needed, such as computing the free energy distance. The code for all preparatory work is provided in 'prepare.ipynb'. 

The method of computing the whole free energy distance matrix is given in 'algorithms/dissim.py'.
The method of computing the free energy distance stated in Section 5.3 is given in 'fedist.py'.



