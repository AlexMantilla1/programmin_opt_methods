from typing import Callable, Dict
import numpy as np
import numpy.typing as npt 
# Import my functionalities:
import sys
sys.path.append("../useful_python_scripts/") 
from golden_section_method_Rn import get_solution_by_golden_section_method
from useful_plots import plot_level_curves_and_points
from derivative_funcitons import gradient_of_fun_in_point
 
"""

"""

def steepest_decent_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float], 
    max_iter: int = 500,
    max_step: float = 10.0,
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

        iterations += 1
        # 1. Exploratory search
        new_point: npt.ArrayLike = last_point.copy()
        # Calculate the gradient in the point
        gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function,new_point)
        # End the algorithm if the magnitude of the gradient is lower that epsilon
        magnitude: float = np.linalg.norm(gradient)
        if magnitude < epsilon:
            break
        # If gradient is large enough define the direction of the gradient
        direction: npt.ArrayLike = (-1.0/magnitude)*gradient
        # Perform golden section search along this direction
        solution: Dict = get_solution_by_golden_section_method(
            epsilon / 4,
            [-max_step, max_step],
            obj_function,
            new_point,
            direction,
        ) 
        # Check if solution was found
        if not solution["converged"]:
            solution["trayectory"] = all_points_trayectory
            return solution
        # Move the point the dir_index dimention by the opt solution found
        new_point = new_point + solution["value"]*direction
        # Add the point to the trayectory
        all_points_trayectory.append(new_point.copy())

        # update for next hooke_and_jeeves iteration
        last_point = new_point.copy()

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
    epsilon: float = 0.001
    # Define an arbitrary point
    # initial_point: npt.ArrayLike = np.array([-2.1, 7.0])
    initial_point: npt.ArrayLike = np.array([0.0, 3.0])
    # The function to minize:
    fun = function_R2 
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: Dict = steepest_decent_method(epsilon, initial_point, fun)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory length
    print(f"Points in trayectory: {len(solution["trayectory"])}")

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main()
