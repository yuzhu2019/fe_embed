## Read network data
## Author: Yu Zhu, Rice ECE, yz126@rice.edu
## 2020 

import numpy as np
import networkx as nx
import csv
import scipy.io

#################################################################
############                Data Files               ############
#################################################################

# file_data = 'data/'
file_data = '/users/yu/Desktop/embed/data/'

'''
  BlogCatalog, Citeseer, Cora, Flickr, YouTube are downloaded from:
    http://zhang18f.myweb.cs.uwindsor.ca/datasets/
'''

file_BlogCatalog = file_data + 'BlogCatalog/edges.csv' # 333983 rows (unweighted)
file_BlogCatalog_label = file_data + 'BlogCatalog/group-edges.csv' # 14476 rows

file_Citeseer = file_data + 'Citeseer/edges.csv' # 4715 rows (unweighted)
file_Citeseer_label = file_data + 'Citeseer/group-edges.csv' # 3312 rows

file_Cora = file_data + 'Cora/edges.csv' # 5429 rows (unweighted)
file_Cora_label = file_data + 'Cora/group-edges.csv' # 2708 rows

file_Flickr = file_data + 'Flickr/edges.csv' # 5899882 rows (unweighted) 
file_Flickr_label = file_data + 'Flickr/group-edges.csv' # 107741 rows

file_YouTube = file_data + 'YouTube/edges.csv' # 2990443 rows (unweighted)
file_YouTube_label = file_data + 'YouTube/group-edges.csv' # 50691 rows

'''
  BrazilAir, EuropeAir, USAir are downloaded from:
    https://github.com/leoribeiro/struc2vec/tree/master/graph
'''

file_BrazilAir = file_data + 'BrazilAir/brazil-airports.edgelist' # (unweighted)
file_BrazilAir_label = file_data + 'BrazilAir/labels-brazil-airports.txt'

file_EuropeAir = file_data + 'EuropeAir/europe-airports.edgelist' # (unweighted)
file_EuropeAir_label = file_data + 'EuropeAir/labels-europe-airports.txt'

file_USAir = file_data + 'USAir/usa-airports.edgelist' # (unweighted)
file_USAir_label = file_data + 'USAir/labels-usa-airports.txt'

'''
  PPI, WikiWord, Facebook, AstroPh are downloaded from:
    https://snap.stanford.edu/node2vec/
'''

file_PPI = file_data + 'PPI/Homo_sapiens.mat' # edgelist and labels (unweighted)

file_WikiWord = file_data + 'WikiWord/POS.mat' # edgelist and labels ( *** weighted *** )

file_Facebook = file_data + 'Facebook/facebook_combined.txt' # edgelist (unweighted)

file_AstroPh = file_data + 'CA-AstroPh/CA-AstroPh.txt' # edgelist (unweighted)

#################################################################
############             Read from Files             ############
#################################################################

def read_edgelist_csv(file_csv):
    nx_G = nx.DiGraph() 
    f = csv.reader(open(file_csv, 'r'))
    for row in f: 
        nx_G.add_edge(row[0], row[1])
    return nx_G

def read_node_attr_csv(nx_G, file_csv):
    node_attr = {node:[] for node in nx_G.nodes()}
    f = csv.reader(open(file_csv, 'r'))
    for row in f:
        if row[0] in node_attr:
            temp = node_attr[row[0]]
            node_attr[row[0]] = temp + [row[1]]
    return node_attr
     
def read_node_attr_txt(nx_G, file_txt, skip):
    node_attr = {node:[] for node in nx_G.nodes()}
    with open(file_txt) as f:
        if skip:
            next(f) # skip the first line
        for line in f:
            (node, label) = line.split()
            if node in node_attr:
                temp = node_attr[node]
                node_attr[node] = temp + [label]
    return node_attr

def read_graph(file_edge, file_label, selfloop=True, connected=False, skip=True):
    
    '''
      - Remove self-loop edges if they exist.
      - For directed graphs, convert to undirected graphs.
      - For a disconnected graph, consider its largest connected component.
    '''
    
    # read edges 
        
    if '.csv' in file_edge:
        G0 = read_edgelist_csv(file_edge)
        for edge in G0.edges():
            G0[edge[0]][edge[1]]['weight'] = 1 
        
    elif '.edgelist' in file_edge:
        G0 = nx.read_edgelist(file_edge, create_using=nx.DiGraph()) 
        for edge in G0.edges():
            G0[edge[0]][edge[1]]['weight'] = 1 
        
    elif '.mat' in file_edge:
        graph_data = scipy.io.loadmat(file_edge)
        sparse_adj = graph_data['network']
        G0 = nx.from_scipy_sparse_matrix(sparse_adj, create_using=nx.DiGraph(), edge_attribute='weight')
    
    print('read as directed: {} nodes, {} edges'.format(len(G0), G0.size()))      
    
    # remove selfloop edges 
    
    if selfloop:
        G0.remove_edges_from(nx.selfloop_edges(G0))
        print('remove selfloop edges: {} nodes, {} edges'.format(len(G0), G0.size())) 

    # convert to undirected graph   
        
    G0 = G0.to_undirected()
    print('convert to undirected: {} nodes, {} edges'.format(len(G0), G0.size())) 

    # check connectivity 
    
    if connected:
        nx_G = G0
    else:
        nx_G = G0.subgraph(max(nx.connected_components(G0), key=len)) 
        print('keep the largest cc: {} nodes, {} edges'.format(len(nx_G), nx_G.size())) 

    # read labels     
        
    if '.csv' in file_label:    
        node_attr = read_node_attr_csv(nx_G, file_label)
        
    elif '.txt' in file_label:
        node_attr = read_node_attr_txt(nx_G, file_label, skip)
        
    elif file_label == ' ': # contained in file_edge
        mat = graph_data['group'].todense()
        node_attr = {i:list(np.where(mat[i]==1)[1]) for i in range(np.size(mat,0))}    
    
    nx.set_node_attributes(nx_G, node_attr, 'label')    
    
    return nx_G

#################################################################
############             Generate Graphs             ############
#################################################################    
    
def gen_graph(label):
        
    if label == 'BlogCatalog': 
        nx_G = read_graph(file_BlogCatalog, file_BlogCatalog_label, False, True)
    
    elif label == 'Citeseer':
        nx_G = read_graph(file_Citeseer, file_Citeseer_label)

    elif label == 'Cora':    
        nx_G = read_graph(file_Cora, file_Cora_label, False)
        
    elif label == 'Flickr':
        nx_G = read_graph(file_Flickr, file_Flickr_label, False, True)
        
    elif label == 'YouTube':    
        nx_G = read_graph(file_YouTube, file_YouTube_label, False)
        
    elif label == 'BrazilAir':
        nx_G = read_graph(file_BrazilAir, file_BrazilAir_label, True, True)
        
    elif label == 'EuropeAir':
        nx_G = read_graph(file_EuropeAir, file_EuropeAir_label, True, True)
        
    elif label == 'USAir':    
        nx_G = read_graph(file_USAir, file_USAir_label, False, False)
        
    elif label == 'PPI':
        nx_G = read_graph(file_PPI, ' ')
        
    elif label == 'WikiWord':
        nx_G = read_graph(file_WikiWord, ' ')
        
    elif label == 'Facebook':
        nx_G = nx.read_edgelist(file_Facebook)
        for edge in nx_G.edges():
            nx_G[edge[0]][edge[1]]['weight'] = 1
        
    elif label == 'AstroPh':
        G0 = nx.read_edgelist(file_AstroPh)
        nx_G = G0.subgraph(max(nx.connected_components(G0), key=len)) 
        for edge in nx_G.edges():
            nx_G[edge[0]][edge[1]]['weight'] = 1
        
    else:
        print('Error! Check the graph name!')
                      
    return nx_G


#################################################################
############           Auxiliary Functions           ############
#################################################################    

def find_labels(nx_G, node_attr):
    labels = set()
    for node in nx_G.nodes(data=node_attr):    
        labels.update(set(node[1])) 
    labels = list(labels)
    n_labels = len(labels)
    return labels, n_labels

