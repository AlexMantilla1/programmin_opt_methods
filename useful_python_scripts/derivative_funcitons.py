from typing import Callable, List
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


def calc_dominant_eigenvalue_of_matrix(
    input_matrix: npt.ArrayLike, tol: float, max_iter: int = 20
) -> float:
    # define arbitrary starting value
    eigenvalue: float = 0.0
    x_vec: npt.ArrayLike = np.array([1.0] + ((len(input_matrix) - 1) * [0.0]))

    while True:
        x_vec_new: npt.ArrayLike = np.matmul(test_matrix, x_vec)
        eigenvalue_new: float = np.dot(x_vec, x_vec_new) / np.dot(x_vec, x_vec)
        if abs(eigenvalue_new - eigenvalue) < tol:
            break
        eigenvalue = eigenvalue_new
        x_vec = x_vec_new.copy()

    return eigenvalue


def calculate_new_D_matrix(
    D_matrix: npt.ArrayLike, p_j: npt.ArrayLike, q_j: npt.ArrayLike
) -> npt.ArrayLike:
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
    first_num = np.outer(p_j, p_j)  # matrix
    first_den = np.dot(p_j, q_j)  # scalar
    first = (1 / first_den) * first_num  # matrix

    # Second term to add
    second_num_a = np.matmul(D_matrix, q_j)  # vector
    second_num_b = np.matmul(q_j, D_matrix)  # vector
    second_num = np.outer(second_num_a, second_num_b)  # matrix
    second_den = np.matmul(second_num_a, q_j)  # scalar
    second = (1 / second_den) * second_num  # matrix

    # Adding to calculate new D matrix
    new_D_matrix = D_matrix + first - second  # matrix

    return new_D_matrix


def derivate_fun_in_point_and_dir(
    fun: Callable[[npt.ArrayLike], float],
    point: npt.ArrayLike,
    dir: npt.ArrayLike,
    epsilon: float = 1e-6,
) -> float:
    """
    Approximate the value of the derivative of a function at a given point in a given direction using central difference method.
    """
    # Check the dir has unitary mag
    if not np.linalg.norm(dir) == 1.0:
        dir = (1 / np.linalg.norm(dir)) * dir

    point_left: npt.ArrayLike = point - (epsilon * dir)
    point_right: npt.ArrayLike = point + (epsilon * dir)

    return (fun(point_right) - fun(point_left)) / (2 * epsilon)


def gradient_of_fun_in_point(
    fun: Callable[[npt.ArrayLike], float],
    point: npt.ArrayLike,
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
        [derivate_fun_in_point_and_dir(fun, point, dir) for dir in directions]
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


if __name__ == "__main__":
    import math

    """ Testing der and grad: """

    def gradent_of_function_R2(x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Example function to test.
        For f: R2 -> R
        """
        x1 = x[0]
        x2 = x[1]
        der_in_x = 2 * (2 * (x1 - 2) ** 3 + x1 - 2 * x2)
        der_in_y = 8 * x2 - 4 * x1
        return np.array([der_in_x, der_in_y])

    def function_R2(x: npt.ArrayLike) -> float:
        """
        Example function to test.
        For f: R2 -> R
        """
        x1 = x[0]
        x2 = x[1]
        return (x1 - 2) ** 4 + (x1 - (2 * x2)) ** 2

    point: npt.ArrayLike = np.array([2.5, 1.5])
    gradiente_1: npt.ArrayLike = gradent_of_function_R2(point)
    print(gradiente_1)
    gradiente_2: npt.ArrayLike = gradient_of_fun_in_point(function_R2, point)
    print(gradiente_2)

    """ Testing hessian matrix: """

    # Example usage:
    def example_function(x):
        return x[0] ** 2 + x[1] ** 2

    # Example usage:
    def example_function_2(x):
        return math.exp(x[0] / 2) * math.sin(x[1])

    point = np.array([2, math.pi / 2])
    hessian = hessian_matrix(example_function_2, point)
    print("Hessian matrix at point {}: \n{}".format(point, hessian))

    """ Testing dominant eigenvalue: """
    print("\n\n\n")
    test_matrix = np.array([[6, 5], [4, 5]])
    dom_eigen: float = calc_dominant_eigenvalue_of_matrix(test_matrix, 0.001)
    print(dom_eigen)
    print("\n\n\n")
    print(np.linalg.cholesky(test_matrix))
