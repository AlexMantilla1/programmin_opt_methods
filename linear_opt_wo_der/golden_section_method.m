%% Clear workspace
clc; clear all; close all;

%% Define the function to optimze (look for minimum)
%f_t = @(t) 2*(t.^2) + 3*t + 7;
f_t = @(t) (t.^2) + 2*t;
t = -3:0.01:5;
f = f_t(t);

% Plot the objective function
fig = figure;
subfigure = subplot(1,1,1);
line1 = plot(t,f,'LineWidth',2); grid on; hold on;
title("Objective Function: f(t) = 2\cdott^2 + 3\cdott + 1.2");
xlabel("t"); ylabel("f(t)");

%% Golden Section Method algorithm

% distinguishability constant
epsilon = 0.0001;
% allowable final length of uncertainty
l = 0.2;
% interval of uncertainty
interval = [-3 5]; 
% alpha parameter
alpha = 0.618;
lambda = interval(1) + (1 - alpha)*(interval(2) - interval(1));
lambda = round(lambda,3);
miu = interval(1) + alpha*(interval(2) - interval(1)); 
miu = round(miu,3);

% counting the steps
n = 0;
% plot the interval
line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none');
scale = abs(interval(1) - interval(2));

while interval(2) - interval(1) > l
    
    % Eval the objective function at the evaluating points
    f1 = f_t(lambda);
    f2 = f_t(miu);
    % Check values to define new interval
    if f1 <= f2 % step 3 from book
        interval = [interval(1) miu];
        % New evaluating points
        miu = lambda;
        lambda = interval(1) + (1 - alpha)*(interval(2) - interval(1));
        lambda = round(lambda,3);
    else %(if f1 > f2) step 2 from book 
        interval = [lambda interval(2)];
        % New evaluating points
        lambda = miu;
        miu = interval(1) + alpha*(interval(2) - interval(1)); 
        miu = round(miu,3); 
    end
    
    n = n+1;  

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
    fprintf("%d: The length of the interva is: %.3f\n",n, interval(2) - interval(1));
end

fprintf("Answer: The optimal value is in the interval [%.3f %.3f]\n",round(interval(1),3),round(interval(2),3));

t_opt = (interval(2) + interval(1)) / 2;
plot(t_opt,f_t(t_opt),'r','Marker','o','MarkerSize',8,'MarkerFaceColor','r');


%% 
