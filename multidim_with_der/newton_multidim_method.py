from typing import Callable, Dict, List
import numpy as np
import numpy.typing as npt 
# Import my functionalities:
import sys
sys.path.append("../useful_python_scripts/") 
from golden_section_method_Rn import get_solution_by_golden_section_method
from useful_plots import plot_level_curves_and_points
from derivative_funcitons import gradient_of_fun_in_point, hessian_matrix
  
def newton_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float], 
    max_iter: int = 500, 
) -> Dict:
    """
    Newton's method for optimizing a multidimensional function.

    Parameters:
        epsilon (float): Convergence criterion. Terminate when the magnitude of the new movement
                         is less than epsilon.
        initial_point (array_like): Initial guess for the optimal solution.
        obj_function (callable): The objective function to minimize.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.

    Returns:
        dict: A dictionary containing:
              - 'value': The optimized solution.
              - 'converged': Boolean indicating if the algorithm converged.
              - 'iterations': Number of iterations performed.
              - 'trayectory': List of all points visited during optimization.
    """

    # Set up
    iterations: int = 0
    last_point: npt.ArrayLike = initial_point.copy()
    all_points_trayectory: List[npt.ArrayLike] = [initial_point]

    while iterations < max_iter: 
        # Define the new_point with initial value
        new_point: npt.ArrayLike = last_point.copy()
        # Calculate the gradient in the point
        gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function, new_point)
        # Calculate the hessian matrix of the fun in the point
        hessian: npt.ArrayLike = hessian_matrix(obj_function, new_point)
        # Calculate the inverse of the Hessian matrix
        inv_hessian: npt.ArrayLike = np.linalg.inv(hessian)
        # Calculate the next movement of the point using the Newton's method formula
        new_mov: npt.ArrayLike = np.matmul(gradient, inv_hessian)
        # End the algorithm if the magnitude of the new movement is lower that epsilon
        magnitude: float = np.linalg.norm(new_mov)
        if magnitude < epsilon:
            break
        # Move the point along the direction found by the Newton's method
        new_point = new_point - new_mov
        # Add the point to the trayectory
        all_points_trayectory.append(new_point.copy()) 
        # Update for next iteration
        last_point = new_point.copy()
        # Increment the iteration count
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
    epsilon: float = 0.001
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
