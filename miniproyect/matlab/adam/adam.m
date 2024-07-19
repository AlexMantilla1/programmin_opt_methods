function solution = adam(epsilon, initial_theta, obj_loss_function, max_iter)

    if nargin < 4
        max_iter = 150;
    end

    % For learning
    step_size = 0.05;       % step size (called "E" in book)
    p1 = 0.9;                % Exponential decay rate p1
    p2 = 0.999;              % Exponential decay rate p2
    sigma = 1e-8;            % Small constant used to stabilize division by small numbers 

    % Initializations for algorithm
    theta = initial_theta;               % initial theta 
    s = zeros(size(theta));            % initial s
    r = zeros(size(theta));            % initial r
    t = 0;                               % time step

    % Numerical Initializations
    grad_magnitude = 1000 * epsilon;      % just to start the loop
    iterations = 0; 
    best_theta = theta;                   % to save the best theta 
    best_obj_loss = obj_loss_function(best_theta);
    
    trayectory = cell(max_iter,1);
    loss_func_trayectory = zeros(size(trayectory));
    % Start loop
    while (grad_magnitude > epsilon) && (iterations < max_iter)
        iterations = iterations + 1; 
        % 1. get minibatch 
        % 2. approximate gradient of loss_function at theta_weird
        fprintf("Iter %d: Calculating gradient",iterations)
        grad = gradient_of_fun_in_point(obj_loss_function, theta, epsilon/10)';
        grad_magnitude = norm(grad);
        % 3. update time
        t = t + 1;
        % 4. Update biased ﬁrst moment estimate 
        s = p1*s + (1-p1)*grad;
        % 5. Update biased second moment estimate
        r = p2*r + ( (1 - p2)*(grad .* grad) );
        % 6. Correct bias in ﬁrst moment
        s_hat = s / (1 - (p1^t));
        % 7. Correct bias in second moment
        r_hat = r / (1 - (p2^t));
        % 8. Compute update: 
        delta_theta = - step_size * ( s_hat ./ (sqrt(r_hat) + sigma) )
        % 9. Apply update
        theta = theta + delta_theta 
        % 10. get the best theta calculated 
        new_obj_loss = obj_loss_function(theta)
        if new_obj_loss < best_obj_loss
            best_theta = theta;  
            best_obj_loss = new_obj_loss;
        end
        % 11. update for new iteration
        trayectory{iterations} = theta;
        loss_func_trayectory(iterations) = new_obj_loss;
        if iterations > max_iter
            iterations = iterations - 1;
            break  
        end  
    end 
    if iterations < max_iter
        converged = true;
    else
        converged = false;
    end
    solution = struct("iterations", iterations, "value", best_theta, "loss_func_value", best_obj_loss,"trayectory", {trayectory}, "converged", converged);
end
 

function gradient = gradient_of_fun_in_point(fun, point, epsilon)
    % Approximate the value of the gradient of a function at a given point using central difference method.
    
    if nargin < 3
        epsilon = 1e-6;
    end
    
    % Define the dimensions
    num_dim = length(point);
    
    % Define a unitary vector in each dimension
    directions = eye(num_dim);
    
    % Calculate the gradient by differentiating in each direction
    gradient = arrayfun(@(i) derivate_fun_in_point_and_dir(fun, point, directions(:, i), epsilon), 1:num_dim);
    % for i = 1:num_dim
    %     gradient(i) = derivate_fun_in_point_and_dir(fun, point, directions(:, i), epsilon);
    % end
end

function d = derivate_fun_in_point_and_dir(fun, point, direction, epsilon)
    % Approximate the value of the derivative of a function at a given point in a given direction using central difference method.
    
    if nargin < 4
        epsilon = 1e-6;
    end
    
    % Check the direction has unitary magnitude
    if norm(direction) ~= 1.0
        direction = (1 / norm(direction)) * direction;
    end
    
    point_left = point - (epsilon * direction);
    point_right = point + (epsilon * direction);
    
    d = (fun(point_right) - fun(point_left)) / (2 * epsilon);
end 