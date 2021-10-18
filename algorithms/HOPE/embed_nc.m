clear;clc;

graph_name = 'Cora';
K = 8;
ratio = 0.95;

dir = '../../mats/Katz_nc/';
file_adj = [dir, 'unweighted_adj/', graph_name, '.mat'];
file_embed = [dir, 'embeddings/', graph_name, '_', num2str(K), '_', num2str(ratio), '.mat'];

load(file_adj); % adj
A = sparse(adj);
beta = ratio/getRadius(A);
[U, V] = embed_main(A, K, beta); % output U
save(file_embed, 'U');












