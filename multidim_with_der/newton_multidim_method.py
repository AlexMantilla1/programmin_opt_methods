from typing import Callable, Dict
import numpy as np
import numpy.typing as npt 
# Import my functionalities:
import sys
sys.path.append("../useful_python_scripts/") 
from golden_section_method_Rn import get_solution_by_golden_section_method
from useful_plots import plot_level_curves_and_points
from derivative_funcitons import gradient_of_fun_in_point, hessian_matrix
 
"""

"""

def newton_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float], 
    max_iter: int = 500, 
) -> Dict:
    """
    Steepest Decent Method.

    Parameters:
    - epsilon (float): The termination scalar to define the threshold of how close is close enough for the solution.
    - initial_point (npt.ArrayLike): The initial point for the algorithm.
    - obj_function (Callable[[npt.ArrayLike], float]): The function we're minimizing.
    - grad_function Callable[[npt.ArrayLike], npt.ArrayLike]: The function that evaluates 
    - num_dim (int): The number of dimensions of the obj_function (in f: R^n -> R, the value of n).
    - max_iter (int): Max number of iterations allowed to converge. Default is 500.
    - max_step (float): Every iteration will have a step in every dimension, this defines the maximum step. Default is 10.0.

    Returns:
    - Dict: A dictionary containing the result of the optimization process, including 'value' (the optimized point), 'converged' (boolean indicating convergence), 'iterations' (number of iterations performed), and 'trayectory' (List of all points explored during the optimization).
    """
    # Set up
    iterations: int = 0
    last_point: npt.ArrayLike = initial_point.copy()
    all_points_trayectory = [initial_point]

    while iterations < max_iter: 
        # Define the new_point with initial value
        new_point: npt.ArrayLike = last_point.copy()
        # Calculate the gradient in the point
        gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function,new_point)
        # Calculate the hessian matrix of the fun in the point
        hessian: npt.ArrayLike = hessian_matrix(obj_function, new_point)
        # Calculate the inverse of that matrix
        inv_hessian: npt.ArrayLike = np.linalg.inv(hessian)
        # Calculate the next movement of the point
        new_mov: npt.ArrayLike = np.matmul(gradient,inv_hessian)
        # End the algorithm if the magnitude of the new movement is lower that epsilon
        magnitude: float = np.linalg.norm(new_mov)
        if magnitude < epsilon:
            break
        # Move the point the dir_index dimention by the opt solution found
        new_point = new_point - new_mov
        # Add the point to the trayectory
        all_points_trayectory.append(new_point.copy()) 
        # update for next hooke_and_jeeves iteration
        last_point = new_point.copy()
        # cont the iteration
        iterations += 1

    # Check if max iterations achieved
    is_good_sol: bool = iterations < max_iter
    return {
        "value": last_point,
        "converged": is_good_sol,
        "iterations": iterations,
        "trayectory": all_points_trayectory,
    }


def function_R2(x: npt.ArrayLike) -> float:
    """
    Example function to test.
    For f: R2 -> R
    """
    x1 = x[0]
    x2 = x[1]
    return (x1 - 2) ** 4 + (x1 - (2 * x2)) ** 2


def function_R2_b(x: npt.ArrayLike) -> float:
    """
    Example function to test.
    For f: R2 -> R
    """
    x1 = x[0]
    x2 = x[1]
    return (x1 - 2) ** 2 + (x2 - 4) ** 2

def main() -> None:
    # Define a termination scalar, to define the threshold of how much is close enough for the solution
    epsilon: float = 0.05
    # Define an arbitrary point
    # initial_point: npt.ArrayLike = np.array([-2.1, 7.0])
    initial_point: npt.ArrayLike = np.array([0.0, 3.0])
    # The function to minize:
    fun = function_R2 
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: Dict = newton_method(epsilon, initial_point, fun)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory length
    print(f"Points in trayectory: {len(solution["trayectory"])}")

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main()
    """grad = np.array([-44, 24])
    #grad = grad[:, np.newaxis]
    print(grad)
    inv_hass = np.array([ [8, 4], [4, 50]])  # Example input matrix
    inv_hass = (1/384)*inv_hass
    print(inv_hass)
    test = np.matmul(grad,inv_hass)
    print(test)"""
