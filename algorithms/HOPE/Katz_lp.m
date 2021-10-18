clear;clc;

graph_name = 'PPI';
ratio = 0.95;

dir = '../../mats/Katz_lp/';

rounds = 10;
beta_list = [];
for round_i = 1:rounds
%     round_i
    file_adj = [dir, 'unweighted_adj/', graph_name, '_', num2str(round_i - 1), '.mat'];
    load(file_adj); % adj
    A = adj;
    beta = ratio / getRadius(sparse(A));
    beta_list = [beta_list, beta];
%     file_katz = [dir, 'katz/', graph_name, '_', num2str(ratio), '_', num2str(round_i - 1), '.mat'];
%     S = inv(eye(size(A,1)) - beta * A) * beta * A;
%     save(file_katz, 'S');
end
