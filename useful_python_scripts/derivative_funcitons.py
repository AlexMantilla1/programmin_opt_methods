from typing import Callable, List
import numpy as np
import numpy.typing as npt


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
