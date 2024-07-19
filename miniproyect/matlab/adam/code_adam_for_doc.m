%% Clean workspace
clc; clear all; close all;

%& Calculate the parameter to model de modulator
config_model

%% Search for optimal solution.
% Define an arbitrary initial point for training 
initial_model = [0.3; % ps1 = pf1
                 0.3; % ps2 = pf2
                 0.8; % ps3 = pf3
                 ];  

% Define the objective function to optimize
obj_function = @run_sim_and_get_SNDR;

% Calculate the model 
open_system('../DS3or.slx')
solution = adam(0.01, initial_model, obj_function);  
disp(solution.value); 
save('adam.mat',"solution")