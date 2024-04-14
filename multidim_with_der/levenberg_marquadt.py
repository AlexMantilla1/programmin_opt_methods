from typing import Callable, Dict, List
import numpy as np
import numpy.typing as npt 
# Import my functionalities:
import sys
sys.path.append("../useful_python_scripts/")  
from useful_plots import plot_level_curves_and_points
from derivative_funcitons import gradient_of_fun_in_point, hessian_matrix
 
"""

"""

def forward_substitution(L, b):
    """
    Solves the linear system Lx = b for x using forward substitution,
    where L is a lower triangular matrix.
    """
    n = L.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x

def backward_substitution(U, b):
    """
    Solves the linear system Ux = b for x using backward substitution,
    where U is an upper triangular matrix.
    """
    n = U.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def levenberg_marquadt(
    epsilon: float,
    initial_point: npt.ArrayLike,
    delta: float,
    obj_function: Callable[[npt.ArrayLike], float],  
    max_iter: int = 500, 
) -> Dict:
    """
    Levenberg-Marquardt optimization algorithm for minimizing a function.

    Parameters:
        epsilon (float): Convergence criterion. Terminate when the magnitude of the new movement
                         is less than epsilon.
        initial_point (array_like): Initial guess for the optimal solution.
        delta (float): Initial value for the parameter delta. (expected to be short)
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
        # Calculate the gradient in the point
        gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function, last_point)
        # Calculate the hessian matrix of the fun in the point
        hessian: npt.ArrayLike = hessian_matrix(obj_function, last_point)

        # Calculate B^-1 = delta*I + H(xk)
        B_m1 = delta*np.eye(len(hessian)) + hessian
        # Cholesky factorization
        L = np.linalg.cholesky(B_m1) 
        # Solve LL' (X_{k+l} - X_k) = -vf(x_k) for X_{k+l}
        Y: npt.ArrayLike = forward_substitution(L, -1*gradient)         #vec
        new_point: npt.ArrayLike = backward_substitution(L.T, Y) + last_point

        # Add the point to the trayectory
        all_points_trayectory.append(new_point.copy())  

        # Compute f(x_{k+1}), f(x_k) and ratio R_k
        f_k = obj_function(last_point)
        f_kl = obj_function(new_point)
        q_k = f_k + np.dot(gradient, new_point - last_point) + 0.5 * np.dot(np.dot((new_point - last_point).T, hessian), new_point - last_point)
        q_kl = f_kl
        R_k = (f_k - f_kl) / (q_k - q_kl)

        # Update ep_k+l based on R_k
        if R_k < 0.25:
            delta *= 4
        elif R_k > 0.75:
            delta /= 2
 
        # Calculate the next movement of the point
        new_mov: npt.ArrayLike = new_point - last_point
        # End the algorithm if the magnitude of the new movement is lower that epsilon
        magnitude: float = np.linalg.norm(new_mov)
        if magnitude < epsilon:
            break 
        # update for next levenberg_marquadt iteration
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
    epsilon: float = 0.001
    # Define an arbitrary point
    # initial_point: npt.ArrayLike = np.array([-2.1, 7.0])
    initial_point: npt.ArrayLike = np.array([0.0, 3.0])
    # The function to minize:
    fun = function_R2 
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: Dict = levenberg_marquadt(epsilon, initial_point, 0.5, fun)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory length
    print(f"Points in trayectory: {len(solution["trayectory"])}")

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main() 
