from typing import Dict, List, Callable
import numpy as np
import numpy.typing as npt


def forward_substitution(L: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
    """
    Solves the linear system Lx = b for x using forward substitution,
    where L is a lower triangular matrix.
    """
    n = L.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return np.array(x)


def backward_substitution(U: npt.ArrayLike, b: npt.ArrayLike) -> npt.ArrayLike:
    """
    Solves the linear system Ux = b for x using backward substitution,
    where U is an upper triangular matrix.
    """
    n = U.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
    return np.array(x)


def derivate_fun_in_point_and_dir(
    fun: Callable[[npt.ArrayLike], float],
    point: npt.ArrayLike,
    direction: npt.ArrayLike,
    epsilon: float = 1e-6,
) -> float:
    """
    Approximate the value of the derivative of a function at a given point in a given direction using central difference method.
    """
    # Check the direction has unitary mag
    if not np.linalg.norm(direction) == 1.0:
        direction = (1 / np.linalg.norm(direction)) * direction

    point_left: npt.ArrayLike = point - (epsilon * direction)
    point_right: npt.ArrayLike = point + (epsilon * direction)

    return (fun(point_right) - fun(point_left)) / (2 * epsilon)


def gradient_of_fun_in_point(
    fun: Callable[[npt.ArrayLike], float], point: npt.ArrayLike, epsilon=1e-6
) -> npt.ArrayLike:
    """
    Approximate the value of the gradient of a function at a given point using central difference method for df.
    """
    # Define de dimentions
    num_dim = len(point)
    # Define a unitary vector in each dimention
    directions: List = list(np.eye(num_dim))
    # Calculate the gradient by derivating in each direction
    gradient = np.array(
        [
            derivate_fun_in_point_and_dir(fun, point, direction, epsilon)
            for direction in directions
        ]
    )
    return gradient


def hessian_matrix(
    fun: Callable[[npt.ArrayLike], float],
    point: npt.ArrayLike,
    epsilon: float = 1e-5,
) -> npt.ArrayLike:
    """
    Approximate numerically the Hessian matrix of a function evaluated at a given point.

    Parameters:
    - fun (Callable[[npt.ArrayLike], float]): The function to calculate the Hessian matrix for.
    - point (npt.ArrayLike): The point at which to calculate the Hessian matrix.
    - epsilon (float): The step size for finite differences. Default is 1e-5.

    Returns:
    - npt.ArrayLike: The Hessian matrix.
    """

    # Get the number of dimensions
    num_dim = len(point)

    # Initialize Hessian matrix
    hess = np.zeros((num_dim, num_dim))

    # Calculate each element of the Hessian matrix using central difference
    for i in range(num_dim):
        for j in range(num_dim):
            # Perturb the point along the i-th and j-th axes
            perturbed_point1 = point.copy()
            perturbed_point1[i] += epsilon
            perturbed_point1[j] += epsilon
            perturbed_point2 = point.copy()
            perturbed_point2[i] += epsilon
            perturbed_point2[j] -= epsilon
            perturbed_point3 = point.copy()
            perturbed_point3[i] -= epsilon
            perturbed_point3[j] += epsilon
            perturbed_point4 = point.copy()
            perturbed_point4[i] -= epsilon
            perturbed_point4[j] -= epsilon

            # Calculate central difference
            hess[i, j] = (
                fun(perturbed_point1)
                - fun(perturbed_point2)
                - fun(perturbed_point3)
                + fun(perturbed_point4)
            ) / (4 * epsilon**2)

    return hess


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
        # cont the iteration
        iterations += 1
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
            # f"delta = {delta}",
        )
        # magnitude: float = np.linalg.norm(new_mov)
        magnitude: float = np.linalg.norm(
            gradient_of_fun_in_point(obj_function, new_point)
        )
        if magnitude < epsilon:
            break
        # update for next levenberg_marquadt iteration
        last_point = new_point.copy()

    # Check if max iterations achieved
    is_good_sol: bool = iterations < max_iter
    return {
        "value": last_point,
        "converged": is_good_sol,
        "iterations": iterations,
        "trayectory": all_points_trayectory,
        "initial_delta": initial_delta,
    }


# Objective function for optimization
def objective_function(test_model: npt.ArrayLike) -> float:
    # TO BE DEFINED AT MATLAB
    pass


# function to test the optimal solution
def test_model(model_to_test: npt.ArrayLike) -> None:
    # TO BE DEFINED AT MATLAB
    pass


def main() -> None:
    # Define an arbitrary initial point for training
    initial_model: npt.ArrayLike = np.array(
        [
            -1.47758376,
            0.63444572,
            -0.03953152,
            -0.02307469,
            -0.0069757,
            0.01976465,
            0.05790899,
            0.01941009,
        ]
    )
    # Define the objective function to optimize
    obj_function = objective_function
    # Define the method used to train the model
    opt_method_function = levenberg_marquadt
    # Calculate the model
    solution: Dict = opt_method_function(
        0.001,
        initial_model,
        obj_function,
    )
    if not solution["converged"]:
        print("WARNING!!! method didn't converge")

    system_model: npt.ArrayLike = solution["value"]
    # Erase trayectory for confort
    solution["trayectory"] = []
    print(solution)
    # Test the model
    test_model(system_model)


if __name__ == "__main__":
    main()
