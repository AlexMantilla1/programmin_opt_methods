from typing import Callable
import numpy as np
import numpy.typing as npt
from golden_section_method_Rn import golden_section_method


def get_solution_by_golden_section_method(
    epsilon: float,
    initial_interval: list[float],
    obj_fun: Callable[[npt.ArrayLike], float],
    x: npt.ArrayLike,
    direction: npt.ArrayLike,
    iter_number: int,
    max_iter: int = 100,
) -> dict:
    solution: dict = golden_section_method(
        epsilon,
        initial_interval,
        obj_fun,
        x,
        direction,
        max_iter,
    )
    # print(solution)
    if not solution["converged"]:
        print(
            f"ERROR! golden section algorithm didn't converge in dir: {direction}",
            f"Last point: {x}",
            f"Solution found: {solution}",
            f"You could check epsilon, max_iter or max_step args (used: {epsilon/4},{max_iter},{initial_interval[1]})",
        )
        return {
            "value": x,
            "converged": False,
            "iterations": iter_number,
        }
    else:
        return solution


def hooke_and_jeeves_method(
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
        # 1. Exploratory search
        new_point: npt.ArrayLike = last_point.copy()
        # Iterate for every dimention to find a new opt point.
        for dir_index in np.arange(num_dim):
            # print(f"1. new_point is: {new_point}")
            # Look through this direction
            solution: dict = get_solution_by_golden_section_method(
                epsilon / 4,
                [-max_step, max_step],
                obj_function,
                new_point,
                directions[dir_index],
                iterations,
            )
            # print(f"2. solution is: {solution}")
            if not solution["converged"]:
                return solution
            # Calculate new point
            new_point[dir_index] += solution["value"]
        # print("")
        # print(f"3. For this Exploratory search new point is: {new_point}")
        # print(f"4. La norma es: {np.linalg.norm(new_point - last_point)}")
        # quit()
        # End the algorithm if the magnitude of the difference is lower that epsilon
        magnitude: float = np.linalg.norm(new_point - last_point)
        if magnitude < epsilon:
            break

        # 2. Pattern search
        # Calculate direction
        direction: np.ArrayLike = (1 / magnitude) * (new_point - last_point)
        solution: dict = get_solution_by_golden_section_method(
            epsilon / 4,
            [-max_step, max_step],
            obj_function,
            new_point,
            direction,
            iterations,
        )
        if not solution["converged"]:
            return solution

        # Calculate new point
        new_point += solution["value"] * direction

        # update for next hooke_and_jeeves iter
        last_point = new_point.copy()

    # Check if max iter achieved
    is_good_sol = iterations < max_iter
    return {
        "value": new_point,
        "converged": is_good_sol,
        "iterations": iterations,
    }


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
    # initial_point: npt.ArrayLike = np.array([-2.1, 7.0])
    initial_point: npt.ArrayLike = np.array([0.0, 3.0])
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: dict = hooke_and_jeeves_method(epsilon, initial_point, function_R2, 2)
    # Just print solution
    print(solution)


if __name__ == "__main__":
    main()
