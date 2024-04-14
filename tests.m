clear; clc;

y_j = [0 3];
y_j1 = [2.70 1.51];

grad_j = [-44 24];
grad_j1 = [0.73 1.28];

p_j = y_j1 - y_j
q_j = grad_j1 - grad_j

%% last D
D = eye(2);

%% First part
%p_j * p_j'

%p_j' * q_j  

first_num = (p_j') * p_j

first_den = p_j * (q_j')

first = first_num / first_den

%% Second part

second_num_a = D * (q_j') 
second_num_b = q_j * D
second_num = second_num_a * second_num_b

second_den = q_j * D * (q_j')

second = second_num / second_den

%% Answer:

D_new = D + first - second

