function solution = levenberg_marquadt(epsilon, initial_point, obj_function, max_iter)
    % Levenberg-Marquardt optimization algorithm for minimizing a function.
    % 
    % Parameters:
    % epsilon (float): Convergence criterion. Terminate when the magnitude of the new movement
    %                  is less than epsilon.
    % initial_point (array_like): Initial guess for the optimal solution.
    % obj_function (callable): The objective function to minimize.
    % max_iter (int, optional): Maximum number of iterations. Defaults to 100.
    % 
    % Returns:
    % struct: A struct containing:
    %   - value: The optimized solution.
    %   - converged: Boolean indicating if the algorithm converged.
    %   - iterations: Number of iterations performed.
    %   - trayectory: List of all points visited during optimization.

    if nargin < 4
        max_iter = 150;
    end

    % Set up
    initial_delta = 0.8;
    delta = initial_delta;
    iterations = 0;
    last_point = initial_point;
    all_points_trayectory = {initial_point};

    while iterations < max_iter
        % Increment the iteration counter
        iterations = iterations + 1;
        % Calculate the gradient at the point
        fprintf("\nIter %d: Calculating gradient",iterations) 
        gradient = gradient_of_fun_in_point(obj_function, last_point,epsilon)';
        magnitude = norm(gradient); 
        % End the algorithm if the magnitude of the new movement is less than epsilon
        fprintf('! |gradient| = %f\n', magnitude);
        if magnitude < epsilon
            break;
        end
        % Calculate the Hessian matrix of the function at the point
        fprintf("\nIter %d: Calculating hessian matrix",iterations)
        hessian = hessian_matrix(obj_function, last_point, epsilon);
        
        while true
            try
                % Calculate B^-1 = delta*I + H(xk)
                B_m1 = delta * eye(length(hessian)) + hessian;
                % Cholesky factorization
                L = chol(B_m1, 'lower');
                break;
            catch
                delta = delta * 4;
            end
        end
        
        % Solve LL' (X_{k+l} - X_k) = -vf(x_k) for X_{k+l} 
        Y = L \ (-gradient);  % vector 
        step = ((L') \ Y);
        step = epsilon*step./norm(step);
        new_point = last_point + step;
        
        % Add the point to the trajectory
        all_points_trayectory{end+1} = new_point;
        
        % Compute f(x_{k+1}), f(x_k) and ratio R_k
        f_k = obj_function(last_point);
        f_kl = obj_function(new_point);
        %fprintf("SNDR(new_point) = %f",f_kl)
        q_k = f_k + dot(gradient, new_point - last_point) + 0.5 * (new_point - last_point)' * hessian * (new_point - last_point);
        q_kl = f_kl;
        R_k = (f_k - f_kl) / (q_k - q_kl);
        
        % Update delta based on R_k
        if R_k < 0.25
            delta = delta * 4;
        elseif R_k > 0.75
            delta = delta / 2;
        end

        % Update for next Levenberg-Marquardt iteration
        last_point = new_point;
    end
    
    % Check if max iterations achieved
    is_good_sol = iterations < max_iter;
    solution = struct('value', last_point, 'converged', is_good_sol, 'iterations', iterations, 'trayectory', {all_points_trayectory}, 'initial_delta', initial_delta);
end

function hess = hessian_matrix(fun, point, epsilon)
    % Approximate numerically the Hessian matrix of a function evaluated at a given point.
    
    if nargin < 3
        epsilon = 1e-5;
    end
    
    % Get the number of dimensions
    num_dim = length(point);
    
    % Initialize Hessian matrix
    hess = zeros(num_dim, num_dim);
    
    % Calculate each element of the Hessian matrix using central difference
    for i = 1:num_dim
        for j = 1:num_dim
            % Perturb the point along the i-th and j-th axes
            perturbed_point1 = point;
            perturbed_point1(i) = perturbed_point1(i) + epsilon;
            perturbed_point1(j) = perturbed_point1(j) + epsilon;
            
            perturbed_point2 = point;
            perturbed_point2(i) = perturbed_point2(i) + epsilon;
            perturbed_point2(j) = perturbed_point2(j) - epsilon;
            
            perturbed_point3 = point;
            perturbed_point3(i) = perturbed_point3(i) - epsilon;
            perturbed_point3(j) = perturbed_point3(j) + epsilon;
            
            perturbed_point4 = point;
            perturbed_point4(i) = perturbed_point4(i) - epsilon;
            perturbed_point4(j) = perturbed_point4(j) - epsilon;
            
            % Calculate central difference
            hess(i, j) = (fun(perturbed_point1) - fun(perturbed_point2) - fun(perturbed_point3) + fun(perturbed_point4)) / (4 * epsilon^2);
        end
    end
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