clear;clc;

graph_name = 'BlogCatalog';
ratio = 0.95;

dir = '../../mats/Katz_nc/';
file_adj = [dir, 'unweighted_adj/', graph_name, '.mat'];
file_katz = [dir, 'katz/', graph_name, '_', num2str(ratio), '.mat'];

load(file_adj); % adj
A = adj;
beta = ratio/getRadius(sparse(A));

S = (eye(size(A,1)) - beta * A) \ (beta * A);
save(file_katz, 'S');