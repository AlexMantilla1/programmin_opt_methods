from typing import Callable
import numpy as np
import numpy.typing as npt


def get_solution_by_golden_section_method(
    epsilon: float,
    initial_interval: list[float],
    obj_fun: Callable[[npt.ArrayLike], float],
    x: npt.ArrayLike,
    direction: npt.ArrayLike,
    max_iter: int = 100,
) -> dict:
    """This function is just a check for the golden section optimization.
    This function checks if the method converged in the expected iteraitons.
    """
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
            f"WARNING! golden section algorithm didn't converge in dir: {direction}",
            f"Last point: {x}",
            f"Solution found: {solution}",
            f"You could check epsilon, max_iter or max_step args (used: {epsilon},{max_iter},{initial_interval[1]})",
        )
        return {
            "value": x,
            "converged": False,
            "iterations": solution["iterations"],
        }
    else:
        return solution


def golden_section_method(
    l: float,
    initial_interval: list[float],
    obj_fun: Callable[[npt.ArrayLike], float],
    x: npt.ArrayLike,
    direction: npt.ArrayLike,
    max_iter: int = 100,
) -> dict:
    """The golden section method:
    l: maximum length of the optimal interval
    initial_interval: initial interval to find the opt solution
    obj_fun: objective funtion to evaluate
    x: x point to solve the obj function f(x + lambda*direction)
    direction: direction vector to solve the obj function f(x + lambda*direction)
    max_iter: maximum number of expected iterations to find the solution.
    """
    # Setting the first interval to iterate
    interval: list[float] = initial_interval.copy()
    # alpha parameter
    alpha: float = 0.618
    # Calculating lambda and miu for first iter.
    lambda_: float = round(interval[0] + (1 - alpha) * (interval[1] - interval[0]), 4)
    miu: float = round(interval[0] + alpha * (interval[1] - interval[0]), 4)
    iterations: int = 0

    while (interval[1] - interval[0] > l) and (iterations < max_iter):
        # Eval the objective function at the evaluating points
        f1 = obj_fun(x + lambda_ * direction)
        f2 = obj_fun(x + miu * direction)

        if f1 <= f2:  # step 3 from book
            interval[1] = miu
            # New evaluating points
            miu = lambda_
            lambda_ = round(interval[0] + (1 - alpha) * (interval[1] - interval[0]), 4)
        else:  # (if f1 > f2) step 2 from book
            interval[0] = lambda_
            # New evaluating points
            lambda_ = miu
            miu = round(interval[0] + alpha * (interval[1] - interval[0]), 4)

        # Count the iteration
        iterations += 1

        # You can uncomment this print to check every iteration.
        # print(
        #   f"{iterations}: The length of the interval is: {round(interval[1] - interval[0],4)}"
        # )

    # Once the interval is small enough, let's set the output.
    sol: float = (interval[1] + interval[0]) / 2
    # Check if max iter achieved
    is_good_sol = iterations < max_iter
    # Check if method didn't reach a limit
    if sol < (initial_interval[1] + initial_interval[0]) / 2:
        is_good_sol = is_good_sol and (sol > initial_interval[0] + l)
    else:
        is_good_sol = is_good_sol and (sol < initial_interval[1] - l)
    # Final solution
    solution: dict = {"value": sol, "converged": is_good_sol, "iterations": iterations}
    return solution
