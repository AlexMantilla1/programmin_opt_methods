%% Clear workspace
clc; clear all; close all;

%% Define the function to optimze (look for minimum)
%f_t = @(t) 2*(t.^2) + 3*t + 7;
%f_t2 = @(t) 4*(t.^3) - 3*(t.^4);    % if t >= 0
f_t1 = @(t) 4*(t.^3) + 3*(t.^4);    % if t < 0

fd1_t1 = @(t) 12*(t.^2) - 12*(t.^3);
fd2_t1 = @(t) 24*(t) - 36*(t.^2);
paso = 0.001;
t = -3:paso:3;
f = f_t1(t);

% Plot the objective function
fig = figure;
subfigure = subplot(1,1,1);
line1 = plot(t,f,'LineWidth',2); grid on; hold on;
title("Objective Function: \theta(\lambda) = 4\cdot\lambda^3 + 3\cdot\lambda^4");
xlabel("λ"); ylabel("θ(λ)");

%% Newton method

% allowable final length of uncertainty
l = 0.2;
% Termination scalar
epsilon = 0.01;
% starting point
lambda = 0.6;
% interval of uncertainty
interval = [min(lambda,-lambda) max(lambda,-lambda)]; 
% max iters
n = 20;
% plot the interval
line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none'); 
%set(subfigure,'NextPlot','replace');
scale = abs(interval(1) - interval(2));
 
% matriz = zeros(n+1,6);
% matriz(1,:) = [1 interval(1) interval(2) lambda f_t(lambda) fd_t(lambda)];

for k = 1:n
    % Calculate new lambda
    new_lambda = lambda - ( (fd1_t1(lambda)) / (fd2_t1(lambda)) );
    if new_lambda < lambda
        interval = [new_lambda lambda]; 
    else
        interval = [lambda new_lambda]; 
    end 

    %matriz(k+1,:) = [k+1 interval(1) interval(2) lambda f_t(lambda) fd_t(lambda)];
    
    % Plot new interval
    pause(1);
    line2.XData = interval;
    new_scale = abs(interval(1) - interval(2));
    % Update scale to keep visual tracking of the interval 
    if new_scale/scale < 0.1 
        hold off;
        t = interval(1)-abs(interval(1))*1.1:paso:interval(2)+abs(interval(2))*1.1;
        f = f_t1(t); 
        line1 = plot(t,f,'LineWidth',2); grid on; hold on;
        line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none');  
        scale = new_scale;
    end

    % Just to check, print the length of the interval
    fprintf("%d: The length of the interva is: %f\n",k, interval(2) - interval(1)); 
    
    % Check to Finish
    if abs( new_lambda - lambda ) < epsilon 
        break;
    end
    lambda = new_lambda;
end

fprintf("Answer: The optimal value is %f\n",new_lambda);

t_opt = new_lambda;
plot(t_opt,f_t1(t_opt),'r','Marker','o','MarkerSize',8,'MarkerFaceColor','r');