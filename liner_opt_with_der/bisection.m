%% Clear workspace
clc; clear all; close all;

%% Define the function to optimze (look for minimum)
%f_t = @(t) 2*(t.^2) + 3*t + 7;
f_t = @(t) (t.^2) + 2*t;
fd_t = @(t) 2*t + 2;
t = -3:0.01:6;
f = f_t(t);

% Plot the objective function
fig = figure;
subfigure = subplot(1,1,1);
line1 = plot(t,f,'LineWidth',2); grid on; hold on;
title("Objective Function: f(t) = 2\cdott^2 + 3\cdott + 1.2");
xlabel("t"); ylabel("f(t)");

%% Dichotomous Serach algorithm

% allowable final length of uncertainty
l = 0.2;
% interval of uncertainty
interval = [-3 6]; 
% counting the steps
n = ceil( log2( (interval(2) - interval(1)) / l) );
% plot the interval
line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none'); 
%set(subfigure,'NextPlot','replace');
scale = abs(interval(1) - interval(2));

lambda = (interval(2) + interval(1))/2;
matriz = zeros(n+1,6);
matriz(1,:) = [1 interval(1) interval(2) lambda f_t(lambda) fd_t(lambda)];

for k = 1:n
    % Calculate lambda y derivate
    lambda = (interval(2) + interval(1))/2;
    fd_t_lambda = fd_t(lambda);
    
    
    if ( fd_t_lambda == 0 ) % The opt point!
        interval = [fd_t_lambda fd_t_lambda];
        break; 
    elseif ( fd_t_lambda > 0 )  % The opt point is at the left
        interval(2) = lambda;
    else % The opt point is at the right
        interval(1) = lambda; 
    end 


    matriz(k+1,:) = [k+1 interval(1) interval(2) lambda f_t(lambda) fd_t(lambda)];
    
    % Plot new interval
    pause(1);
    line2.XData = interval;
    new_scale = abs(interval(1) - interval(2));
    % Update scale to keep visual tracking of the interval 
    if new_scale/scale < 0.1 
        hold off;
        t = interval(1)-abs(interval(1))*1.1:0.01:interval(2)+abs(interval(2))*1.1;
        f = f_t(t); 
        line1 = plot(t,f,'LineWidth',2); grid on; hold on;
        line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none');  
        scale = new_scale;
    end

    % Just to check, print the length of the interval
    fprintf("%d: The length of the interva is: %f\n",k, interval(2) - interval(1));
end


% Print info and answer
datos_table = array2table(matriz,'VariableNames',{'k' 'a_k' 'b_k' 'lambda' 'theta(lambda_k)' 'theta_d(lambda_k)'});
disp(datos_table)

fprintf("Answer: The optimal value is in the interval [%f %f]\n",interval(1),interval(2));

t_opt = (interval(2) + interval(1)) / 2;
plot(t_opt,f_t(t_opt),'r','Marker','o','MarkerSize',8,'MarkerFaceColor','r');