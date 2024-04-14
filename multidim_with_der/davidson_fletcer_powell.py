from typing import Callable, Dict, List
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

def calculate_new_D_matrix(D_matrix: npt.ArrayLike, p_j: npt.ArrayLike, q_j: npt.ArrayLike) -> npt.ArrayLike:
    """
    Calculate the updated D matrix for the Davidson-Fletcher-Powell (DFP) multivariable optimization method.

    Parameters:
    - D_matrix (array-like): The current D matrix, a symmetric positive definite matrix.
    - p_j (array-like): The change in the parameters vector between iterations.
    - q_j (array-like): The change in the gradient vector between iterations.

    Returns:
    - new_D_matrix (array-like): The updated D matrix after applying the DFP method.
    """

    # First term to add
    first_num = np.outer(p_j, p_j)        # matrix
    first_den = np.dot(p_j, q_j)          # scalar
    first = (1 / first_den) * first_num   # matrix

    # Second term to add
    second_num_a = np.matmul(D_matrix, q_j)          # vector
    second_num_b = np.matmul(q_j, D_matrix)          # vector
    second_num = np.outer(second_num_a, second_num_b) # matrix
    second_den = np.matmul(second_num_a, q_j)          # scalar
    second = (1 / second_den) * second_num         # matrix

    # Adding to calculate new D matrix
    new_D_matrix = D_matrix + first - second   # matrix

    return new_D_matrix
 

def davidson_fletcer_powell(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float], 
    initial_D_matrix: npt.ArrayLike,
    max_iter: int = 500, 
    max_step: float = 50.0,
) -> Dict:
    """
    Davidson-Fletcher-Powell optimization algorithm.

    Parameters:
        epsilon (float): Convergence criterion. Terminate when the magnitude of the gradient
                         is less than epsilon.
        initial_point (array_like): Initial guess for the optimal solution.
        obj_function (callable): The objective function to minimize.
        initial_D_matrix (array_like): Initial value for the D matrix.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        max_step (float, optional): Maximum step size for the golden section search. Defaults to 50.0.

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
    
    Rn: int = len(initial_point)

    while iterations < max_iter: 
        
        # D_matrix initialized every iteration to the initial value.
        D_matrix: npt.ArrayLike = initial_D_matrix.copy()
        for j in np.arange(Rn):
            # Define the new_point with initial value
            new_point: npt.ArrayLike = last_point.copy()
            # Calculate the gradient in the point
            gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function, new_point) 
            # End the algorithm if the magnitude of the gradient is lower that epsilon
            magnitude: float = np.linalg.norm(gradient)
            if magnitude < epsilon:
                break

            # Calculate the new direction of the movement:
            new_dir: npt.ArrayLike = -1 * np.matmul(gradient, D_matrix)
            # Perform golden section search along this direction
            golden_search_solution: Dict = get_solution_by_golden_section_method(
                epsilon / 4,
                [0, max_step],
                obj_function,
                new_point,
                new_dir,
            ) 
            # Check if golden_search_solution was found
            if not golden_search_solution["converged"]:
                return {
                    "value": new_point,
                    "converged": False,
                    "iterations": iterations,
                    "trayectory": all_points_trayectory,
                }

            # Move the point in the calculated direction by the size of the golden_search_solution found
            new_point = new_point + golden_search_solution["value"] * new_dir
    
            # Add the point to the trayectory
            all_points_trayectory.append(new_point.copy())  
            # Update for next iteration
            last_point = new_point.copy()

            # Update D_matrix
            if j < Rn:
                p_j: npt.ArrayLike = golden_search_solution["value"] * new_dir
                q_j: npt.ArrayLike = gradient_of_fun_in_point(obj_function, new_point) - gradient
                D_matrix = calculate_new_D_matrix(D_matrix, p_j, q_j) 

        # End the algorithm if the magnitude of the gradient is lower that epsilon
        if magnitude < epsilon:
            break

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
    # initial D matrix
    initial_D_matrix = np.eye(2)
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: Dict = davidson_fletcer_powell(epsilon, initial_point, fun, initial_D_matrix)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory length
    print(f"Points in trayectory: {len(solution["trayectory"])}")
    # Print the trayectory length
    """for point in solution["trayectory"]:
        print(point)"""

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main()  
