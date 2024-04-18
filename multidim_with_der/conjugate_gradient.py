from typing import Callable, Dict, List
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

def conjugate_gradient(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float], 
    max_iter: int = 500,
    max_step: float = 50.0,
) -> Dict: 
    """
    Conjugate Gradient optimization algorithm.

    Parameters:
        epsilon (float): Convergence criterion. Terminate when the magnitude of the gradient
                         is less than epsilon.
        initial_point (array_like): Initial guess for the optimal solution.
        obj_function (callable): The objective function to minimize.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        max_step (float, optional): Maximum step size for the golden section search. Defaults to 10.0.

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
    new_point: npt.ArrayLike = last_point.copy()
    Rn: int = len(initial_point)
    direction: npt.ArrayLike
    last_direction: npt.ArrayLike
    gradient: npt.ArrayLike
    last_gradient: npt.ArrayLike


    while iterations < max_iter:

        iterations += 1 
        for j in np.arange(Rn):
            # Calculate the gradient in the point
            gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function, last_point)
            # End the algorithm if the magnitude of the gradient is lower that epsilon
            magnitude: float = np.linalg.norm(gradient)
            if magnitude < epsilon:
                break

            # Check if alpha needed
            if j == 0: 
                # If gradient is large enough, define the direction of the gradient
                direction = (-1.0) * gradient
            else:
                alpha = (magnitude**2) / (np.linalg.norm(last_gradient)**2)
                direction = (-1.0) * gradient + (alpha*last_direction)


            # Perform golden section search along this direction
            solution: Dict = get_solution_by_golden_section_method(
                epsilon / 4,
                [0.0, max_step],
                obj_function,
                last_point,
                direction,
            ) 
            # Check if solution was found
            if not solution["converged"]:
                solution["trayectory"] = all_points_trayectory
                return solution
            # Move the point along the direction found by the golden section search
            new_point = last_point + solution["value"] * direction
            # Add the point to the trajectory
            all_points_trayectory.append(new_point.copy()) 
            # Update for next iteration
            last_point = new_point.copy()
            last_direction = direction
            last_gradient = gradient

        if magnitude < epsilon:
            break  

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
    solution: Dict = conjugate_gradient(epsilon, initial_point, fun)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory length
    print(f"Points in trayectory: {len(solution["trayectory"])}")
    print(solution)

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main()
