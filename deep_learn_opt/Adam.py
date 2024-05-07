from typing import Callable, Dict, List
import numpy as np
import pandas as pd

# Import my functionalities:
import sys

sys.path.append("../useful_python_scripts/")
from useful_plots import plot_model_comparison
from derivative_funcitons import gradient_of_fun_in_point


def get_data() -> List[List]:
    # Read the CSV file
    df = pd.read_csv("data.csv")
    # Extract each column as a vector
    input_signal = df["input"].tolist()
    output_signal = df["output"].tolist()
    # Return the data
    del df
    return [input_signal, output_signal]


def evaluate_output_using_model(
    system_model: List[float], input_signal: List[float], initial_condition: float
) -> List[float]:
    # get the order
    order: int = int(len(system_model) / 2)
    system_model = np.array(system_model)
    # First "order" values assume steady state with initial condition
    extended_output: List[float] = order * [initial_condition]
    extended_input: List[float] = (order * [input_signal[0]]) + input_signal
    for n in np.arange(order, len(extended_input)):
        # Create the window that affects the output at n
        interest_window: List[float] = []
        for i in np.arange(order):
            interest_window.append(-extended_output[n - i - 1])
        for i in np.arange(order):
            interest_window.append(extended_input[n - i - 1])

        # The n-th value of the output is
        extended_output.append(np.dot(system_model, interest_window))

    return extended_output[order:]


def calculate_cuadratic_mean_error(
    output_signal_model: List[float], real_output: List[float]
) -> float:
    cuadratic_mean_error: float = 0.0
    N: int = len(real_output)
    # Acumulate the Cuadratic mean error
    for n in np.arange(N):
        cuadratic_mean_error = cuadratic_mean_error + (
            ((real_output[n] - output_signal_model[n]) ** 2) / N
        )
    # Return the cuadratic mean error
    return cuadratic_mean_error


def loss_function(model: List[float]) -> float:
    # Evaluate output with model
    global x_data, y_data
    output_signal_model = evaluate_output_using_model(model, x_data, y_data[0])
    # Return the cuadratic mean error
    return calculate_cuadratic_mean_error(output_signal_model, y_data)


def Adam_method(
    epsilon: float,
    initial_theta: List[float],
    obj_loss_function: Callable[[List[float]], float],
    max_iter: int = 100,
) -> Dict:
    # For learning
    step_size = 0.001       # step size (called "E" in book)
    p1 = 0.9                # Exponential decay rate p1
    p2 = 0.999              # Exponential decay rate p2
    sigma = 1e-8            # Small constant used to stabilize division by small numbers  

    # Initializations for algorithm
    theta = np.array(initial_theta)     # initial theta 
    s = np.zeros(len(theta))            # initial s
    r = np.zeros(len(theta))            # initial r
    t: int = 0                          # time step

    # Numerical Initializations
    grad_magnitude: float = 1000 * epsilon      # just to start the loop
    iterations = 0 
    best_theta = np.copy(theta)         # to save the best theta
    mini_batch_size: int = 200
    num_batches: int = int(len(input_signal) / mini_batch_size)  
    global x_data, y_data
    
    # Start loop
    while (grad_magnitude > epsilon) and (iterations < max_iter):
        iterations = iterations + 1
        for k in np.arange(num_batches):
            # 1. get minibatch
            x_data = input_signal[k * mini_batch_size : (k + 1) * mini_batch_size]
            y_data = output_signal[k * mini_batch_size : (k + 1) * mini_batch_size] 
            # 2. approximate gradient of loss_function at theta_weird
            grad = gradient_of_fun_in_point(obj_loss_function, theta, epsilon/10) 
            grad_magnitude = np.linalg.norm(grad)
            # 3. update time
            t = t + 1
            # 4. Update biased ﬁrst moment estimate 
            s = p1*s + (1-p1)*grad
            # 5. Update biased second moment estimate
            r = p2*r + ( (1 - p2)*(grad * grad) )
            # 6. Correct bias in ﬁrst moment
            s_hat = s / (1 - (p1**t))
            # 7. Correct bias in second moment
            r_hat = r / (1 - (p2**t))
            # 8. Compute update: 
            delta_theta = - step_size * ( s_hat / (np.sqrt(r_hat) + sigma) )
            # 9. Apply update
            theta = theta + delta_theta 
            # 10. get the best theta calculated
            x_data = input_signal.copy()
            y_data = output_signal.copy() 
            if obj_loss_function(theta) < obj_loss_function(best_theta):
                best_theta = np.copy(theta) 
            # 11. update for new iteration
            iterations = iterations + 1
            if iterations > max_iter:
                iterations = iterations - 1
                break 

    return {"value": list(best_theta), "iterations": iterations}


def main() -> None:
    # Define initial model
    # initial_model = [-1.50679337, 0.57994944, -0.04300483, 0.11451339]
    # initial_model = [-1.506, 0.579, -0.043, 0.114]
    initial_model = [-1.4, 0.6, -0.1, 0.2]
    # initial_model = [0.1, 0.1, -0.1, -0.1]
    print(f"initial model: {initial_model}")
    # Call function to run Stochastic gradient descent (SGD)
    solution: Dict = Adam_method(
        0.001, initial_model, loss_function, 2000
    )
    sol = solution["value"]
    print(f"Solution achieved: {sol}") 
    print(f"function at that value: {loss_function(sol)}. Itereations required: {solution["iterations"]}")

    output_signal_model = evaluate_output_using_model(
        sol, input_signal, output_signal[0]
    )

    # plot the model to compare
    plot_model_comparison(input_signal, output_signal, output_signal_model)


if __name__ == "__main__":
    # Get data to compare
    input_signal, output_signal = get_data()
    # for minibatches
    x_data = input_signal
    y_data = output_signal
    # Run the main function
    main()
