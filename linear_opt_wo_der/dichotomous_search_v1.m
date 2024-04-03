%% Clear workspace
clc; clear all; close all;

%% Define the function to optimze (look for minimum)
%f_t = @(t) 2*(t.^2) + 3*t + 7;
t = -10:0.01:10;
f = f_t(t);

% Plot the objective function
fig = figure;
subfigure = subplot(1,1,1);
line1 = plot(t,f,'LineWidth',2); grid on; hold on;
title("Objective Function: f(t) = 2\cdott^2 + 3\cdott + 1.2");
xlabel("t"); ylabel("f(t)");

%% Dichotomous Serach algorithm

% distinguishability constant
epsilon = 0.0001;
% allowable final length of uncertainty
l = 0.01;
% interval of uncertainty
interval = [-10 10]; 
% counting the steps
n = 0;
% plot the interval
line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none'); 
scale = abs(interval(1) - interval(2));

while interval(2) - interval(1) > l
    
    
    n = n+1;
    % interval middle point
    mp = ( interval(2) + interval(1) ) / 2;
    % New evaluating points
    lambda = mp - epsilon;
    miu = mp + epsilon;
    % Eval the objective function at the evaluating points
    f1 = f_t(lambda);
    f2 = f_t(miu);
    % Check values to define new interval
    if f1 < f2
        interval = [interval(1) miu];
    elseif f1 > f2
        interval = [lambda, interval(2)];
    else
        interval = [lambda, miu];
    end
    
    % Plot new interval
    pause(1);
    line2.XData = interval;
    new_scale = abs(interval(1) - interval(2));
    % Update scale to keep visual tracking of the interval.
    if new_scale/scale < 0.1 
        hold off;
        t = interval(1)-abs(interval(1))*1.1:0.01:interval(2)+abs(interval(2))*1.1;
        f = f_t(t); 
        line1 = plot(t,f,'LineWidth',2); grid on; hold on;
        line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none');  
        scale = new_scale;
    end

    % Just to check, print the length of the interval
    fprintf("%d: The length of the interva is: %f\n",n, interval(2) - interval(1));
end

fprintf("Answer: The optimal value is in the interval [%f %f]\n",interval(1),interval(2));

t_opt = (interval(2) + interval(1)) / 2;
plot(t_opt,f_t(t_opt),'r','Marker','o','MarkerSize',8,'MarkerFaceColor','r');