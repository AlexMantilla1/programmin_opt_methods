from typing import Callable, Dict, List
import numpy as np
import numpy.typing as npt

# Import my functionalities:
from golden_section_method_Rn import get_solution_by_golden_section_method
from derivative_funcitons import (
    gradient_of_fun_in_point,
    hessian_matrix,
    calculate_new_D_matrix,
    forward_substitution,
    backward_substitution,
    calculate_new_D_matrix_BFGS,
)


def steepest_decent_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    max_iter: int = 100,
) -> Dict:
    """
    Steepest descent optimization algorithm.

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
    max_step: float = 50.0
    last_point: npt.ArrayLike = initial_point.copy()
    all_points_trayectory: List[npt.ArrayLike] = [initial_point]

    while iterations < max_iter:

        iterations += 1
        # Exploratory search
        new_point: npt.ArrayLike = last_point.copy()
        # Calculate the gradient in the point
        gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function, new_point)
        # End the algorithm if the magnitude of the gradient is lower that epsilon
        magnitude: float = np.linalg.norm(gradient)
        if magnitude < epsilon:
            break
        # If gradient is large enough, define the direction of the gradient
        direction: npt.ArrayLike = (-1.0 / magnitude) * gradient
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
        # Move the point along the direction found by the golden section search
        new_point = new_point + solution["value"] * direction
        # Add the point to the trajectory
        all_points_trayectory.append(new_point.copy())

        # Update for next iteration
        last_point = new_point.copy()

    # Check if max iterations achieved
    is_good_sol: bool = iterations < max_iter
    return {
        "value": last_point,
        "converged": is_good_sol,
        "iterations": iterations,
        "trayectory": all_points_trayectory,
    }


def newton_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    max_iter: int = 100,
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


def davidson_fletcer_powell(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    max_iter: int = 100,
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
    initial_D_matrix: npt.ArrayLike = np.eye(len(initial_point))
    max_step: float = 50.0
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
                q_j: npt.ArrayLike = (
                    gradient_of_fun_in_point(obj_function, new_point) - gradient
                )
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


def levenberg_marquadt(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    max_iter: int = 100,
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
    initial_delta: float = 0.3
    delta: float = initial_delta
    iterations: int = 0
    last_point: npt.ArrayLike = initial_point.copy()
    all_points_trayectory: List[npt.ArrayLike] = [initial_point]

    while iterations < max_iter:
        # Calculate the gradient in the point
        gradient: npt.ArrayLike = gradient_of_fun_in_point(obj_function, last_point)
        # Calculate the hessian matrix of the fun in the point
        hessian: npt.ArrayLike = hessian_matrix(obj_function, last_point)
        while True:
            try:
                # Calculate B^-1 = delta*I + H(xk)
                B_m1 = delta * np.eye(len(hessian)) + hessian
                # Cholesky factorization
                L = np.linalg.cholesky(B_m1)
                break
            except np.linalg.LinAlgError:
                delta *= 4
        # print(delta)
        # Solve LL' (X_{k+l} - X_k) = -vf(x_k) for X_{k+l}
        Y: npt.ArrayLike = forward_substitution(L, -1 * gradient)  # vec
        new_point: npt.ArrayLike = backward_substitution(L.T, Y) + last_point

        # Add the point to the trayectory
        all_points_trayectory.append(new_point.copy())

        # Compute f(x_{k+1}), f(x_k) and ratio R_k
        f_k = obj_function(last_point)
        f_kl = obj_function(new_point)
        q_k = (
            f_k
            + np.dot(gradient, new_point - last_point)
            + 0.5
            * np.dot(
                np.dot((new_point - last_point).T, hessian), new_point - last_point
            )
        )
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
        print(
            f"{iterations}: |gradient| = {np.linalg.norm(gradient_of_fun_in_point(obj_function, new_point))}\n",
            f"delta = {delta}",
        )
        # magnitude: float = np.linalg.norm(new_mov)
        magnitude: float = np.linalg.norm(
            gradient_of_fun_in_point(obj_function, new_point)
        )
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
        "initial_delta": initial_delta,
    }


def broyden_fletcher_goldfarb_shanno(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    max_iter: int = 500,
    max_step: float = 50.0,
) -> Dict:
    """
    Broyden-Fletcher-Goldfarb-Shanno optimization algorithm.

    Parameters:
        epsilon (float): Convergence criterion. Terminate when the magnitude of the gradient
                         is less than epsilon.
        initial_point (array_like): Initial guess for the optimal solution.
        obj_function (callable): The objective function to minimize.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        max_step (float, optional): Maximum step size for the golden section search. Defaults to 50.0.

    Returns:
        dict: A dictionary containing:
              - 'value': The optimized solution.
              - 'converged': Boolean indicating if the algorithm converged.
              - 'iterations': Number of iterations performed.
              - 'trayectory': List of all points visited during optimization.
    """

    # initial D matrix
    initial_D_matrix = np.eye(2)
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
                q_j: npt.ArrayLike = (
                    gradient_of_fun_in_point(obj_function, new_point) - gradient
                )
                D_matrix = calculate_new_D_matrix_BFGS(D_matrix, p_j, q_j)

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
