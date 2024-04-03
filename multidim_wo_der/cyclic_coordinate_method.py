from typing import Callable
import numpy as np
import numpy.typing as npt
from golden_section_method_Rn import golden_section_method


def cyclic_coordinate_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    num_dim: int,
    max_iter: int = 500,
    max_step: float = 10.0,
) -> dict:
    """
    epsilon:        Is the termination scalar, to define the threshold of how much
                        is close enough for the solution.
    initial_point:  Is the initial point for the algorithm
    obj_function:   Is the function we're minimizing
    num_dim:        Is the number of dimentions of the obj_funtions (in f: R^n -> R, the value of n)
    max_iter:       Max number of iterations allowed to converge.
    max_step:       Every iteration will have a step in every dim, this defines the maximum step.
    """
    iterations: int = 0
    last_point: npt.ArrayLike = initial_point.copy()
    directions: npt.ArrayLike = np.eye(num_dim)

    while iterations < max_iter:

        iterations += 1
        # print(f"{iterations}: Point to use: {last_point}")
        new_point: npt.ArrayLike = last_point.copy()

        # Iterate for every dimention to find a new opt point.
        for dir_index in np.arange(num_dim):
            # print(new_point)
            # Look through this direction
            solution: dict = golden_section_method(
                epsilon / 4,
                [-max_step, max_step],
                obj_function,
                new_point,
                directions[dir_index],
            )
            # print(solution)
            if not solution["converged"]:
                print(
                    f"ERROR! golden section algorithm didn't converge in dir: {directions[dir_index]}",
                    f"Last point: (X={new_point[0]},Y={new_point[1]})",
                    f"Solution found: {solution}",
                    f"You could check epsilon, max_iter or max_step args (used: {epsilon/4},{max_iter},{max_step})",
                )
                return {
                    "value": initial_point,
                    "converged": False,
                    "iterations": iterations,
                }

            # Calculate new point
            new_point[dir_index] += solution["value"]
            # update for next dir

        # print(f"La norma es: {np.linalg.norm(new_point - last_point)}")
        if np.linalg.norm(new_point - last_point) < epsilon:
            break

        # update for next iter
        last_point = new_point.copy()

    # Check if max iter achieved
    is_good_sol = iterations < max_iter
    return {
        "value": new_point,
        "converged": is_good_sol,
        "iterations": iterations,
    }


def objective_function(x: list[float]) -> float:
    """For f: R -> R"""
    return (x[0] ** 2) + 2 * x[0]


def function_R2(x: npt.ArrayLike) -> float:
    """For f: R2 -> R"""
    x1 = x[0]
    x2 = x[1]
    return (x1 - 2) ** 4 + (x1 - (2 * x2)) ** 2


def function_R2_b(x: npt.ArrayLike) -> float:
    """For f: R2 -> R"""
    x1 = x[0]
    x2 = x[1]
    return (x1 - 2) ** 2 + (x2 - 4) ** 2


def main() -> None:
    # Define a termination scalar, to define the threshold of how much is close enough for the solution
    epsilon: float = 0.001
    # Define an arbitrary point
    initial_point: npt.ArrayLike = np.array([-2.1, 7.0])
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: dict = cyclic_coordinate_method(epsilon, initial_point, function_R2_b, 2)
    # Just print solution
    print(solution)


if __name__ == "__main__":
    main()
