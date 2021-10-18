clear;clc;

graph_name = 'BlogCatalog';
K = 128;
ratio = 0.95;

dir = '../../mats/Katz_lp_0.2/';

rounds = 10;
for round_i = 1:rounds
    round_i
    file_adj = [dir, 'unweighted_adj/', graph_name, '_', num2str(round_i - 1), '.mat'];
    load(file_adj); % adj
    A = sparse(adj);
    beta = ratio / getRadius(A);
    file_embed = [dir, 'embeddings/', graph_name, '_', num2str(K), '_', num2str(ratio), '_', num2str(round_i - 1), '.mat'];
    [U, V] = embed_main(A, K, beta); % output U
    save(file_embed, 'U');
end




 







    










