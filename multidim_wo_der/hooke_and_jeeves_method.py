from typing import Callable
import numpy as np
import numpy.typing as npt
from golden_section_method_Rn import get_solution_by_golden_section_method
from useful_plots import plot_level_curves_and_points

"""
The Hooke and Jeeves method is a derivative-free optimization algorithm used for unconstrained optimization
problems. It explores the search space by systematically adjusting points in the direction of descent until
a local minimum is found or a termination criterion is met. This method is particularly useful when the 
objective function is not smooth or has discontinuities, making traditional gradient-based methods less 
effective.
"""

def hooke_and_jeeves_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    num_dim: int,
    max_iter: int = 500,
    max_step: float = 10.0,
) -> dict:
    """
    Hooke and Jeeves Method.

    Parameters:
    - epsilon (float): The termination scalar to define the threshold of how close is close enough for the solution.
    - initial_point (npt.ArrayLike): The initial point for the algorithm.
    - obj_function (Callable[[npt.ArrayLike], float]): The function we're minimizing.
    - num_dim (int): The number of dimensions of the obj_function (in f: R^n -> R, the value of n).
    - max_iter (int): Max number of iterations allowed to converge. Default is 500.
    - max_step (float): Every iteration will have a step in every dimension, this defines the maximum step. Default is 10.0.

    Returns:
    - dict: A dictionary containing the result of the optimization process, including 'value' (the optimized point), 'converged' (boolean indicating convergence), 'iterations' (number of iterations performed), and 'trayectory' (list of all points explored during the optimization).
    """
    # Set up
    iterations: int = 0
    last_point: npt.ArrayLike = initial_point.copy()
    directions: npt.ArrayLike = np.eye(num_dim)
    all_points_trayectory = [initial_point]

    while iterations < max_iter:

        iterations += 1
        # 1. Exploratory search
        new_point: npt.ArrayLike = last_point.copy()
        # Iterate for every dimention to find a new opt point.
        for dir_index in np.arange(num_dim): 
            # Perform golden section search along this direction
            solution: dict = get_solution_by_golden_section_method(
                epsilon / 4,
                [-max_step, max_step],
                obj_function,
                new_point,
                directions[dir_index],
            )
            # Check if solution was found
            if not solution["converged"]:
                solution["trayectory"] = all_points_trayectory
                return solution
            # Move the point the dir_index dimention by the opt solution found
            new_point[dir_index] += solution["value"]
            # Add the point to the trayectory
            all_points_trayectory.append(new_point.copy())
            
        # End the algorithm if the magnitude of the difference is lower that epsilon
        magnitude: float = np.linalg.norm(new_point - last_point)
        if magnitude < epsilon:
            break

        # 2. Pattern search
        # Calculate the direction (unitary vector)
        direction: np.ArrayLike = (1 / magnitude) * (new_point - last_point)
        # Perform golden section search along the new direction
        solution: dict = get_solution_by_golden_section_method(
            epsilon / 4,
            [-max_step, max_step],
            obj_function,
            new_point,
            direction,
        )
        # Check if solution was found
        if not solution["converged"]:
            return solution

        # Move the point the new direction by the opt solution found
        new_point += solution["value"] * direction
        all_points_trayectory.append(new_point.copy())

        # update for next hooke_and_jeeves iteration
        last_point = new_point.copy()

    # Check if max iterations achieved
    is_good_sol: bool = iterations < max_iter
    return {
        "value": new_point,
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
    solution: dict = hooke_and_jeeves_method(epsilon, initial_point, fun, 2)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory len
    print(f"Points in trayectory: {len(solution["trayectory"])}")

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main()
