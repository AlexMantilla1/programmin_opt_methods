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

%% Fibonacci Search Method algorithm

% distinguishability constant
epsilon = 0.01;
% allowable final length of uncertainty
l = 0.2;
% interval of uncertainty
interval = [-3 5]; 

% Choosing the Number of Observations
n = 1;
Fib_n = fibonacci(n);
while Fib_n < ( interval(2) - interval(1) ) / l
    n = n + 1;
    Fib_n = fibonacci(n);
end 

% First iter
lambda = interval(1) + ( ( fibonacci(n-2)/Fib_n ) * ( interval(2) - interval(1) ) );
miu = interval(1) + ( ( fibonacci(n-1)/Fib_n ) * ( interval(2) - interval(1) ) );

% plot the interval
line2 = plot(interval,[0 0],'b','Marker','o','MarkerSize',8,'MarkerFaceColor','b','LineStyle','none');
scale = abs(interval(1) - interval(2));
matriz = zeros(9,7);
matriz(1,:) = [1 interval(1) interval(2) lambda miu f_t(lambda) f_t(miu)];

for k = 1:n-2
    if f_t(lambda) > f_t(miu) % step 2
        interval(1) = lambda;
        lambda = miu;
        miu = interval(1) + ( ( fibonacci(n-k-1)/fibonacci(n-k) ) * ( interval(2) - interval(1) ) ); 
    else %step 3
        interval(2) = miu;
        miu = lambda;
        lambda = interval(1) + ( ( fibonacci(n-k-2)/fibonacci(n-k) ) * ( interval(2) - interval(1) ) );
    end

    if k == n-2 % step 5
        miu = lambda + epsilon;
        if f_t(lambda) > f_t(miu)
            interval(1) = lambda;
        else
            interval(2) = miu;
        end
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
    matriz(k+1,:) = [k+1 interval(1) interval(2) lambda miu f_t(lambda) f_t(miu)];

    % Just to check, print the length of the interval
    fprintf("%d: The length of the interva is: %.3f\n",k, interval(2) - interval(1));
end

% Print info and answer
datos_table = array2table(matriz,'VariableNames',{'k' 'a_k' 'b_k' 'lambda_k' 'miu_k' 'theta(lambda_k)' 'theta(miu_k)'});
disp(datos_table)

fprintf("Answer: The optimal value is in the interval [%.3f %.3f]\n",round(interval(1),3),round(interval(2),3));
fprintf("The middle point is %.6f\n",(interval(1)+interval(2))/2);

t_opt = (interval(2) + interval(1)) / 2;
plot(t_opt,f_t(t_opt),'r','Marker','o','MarkerSize',8,'MarkerFaceColor','r');