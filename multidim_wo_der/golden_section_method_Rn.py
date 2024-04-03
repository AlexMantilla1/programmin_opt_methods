from typing import Callable
import numpy as np
import numpy.typing as npt


def golden_section_method(
    l: float,
    initial_interval: list[float],
    obj_fun: Callable[[npt.ArrayLike], float],
    x: npt.ArrayLike,
    direction: npt.ArrayLike,
    max_iter: int = 100,
) -> dict:
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

        # print(
        #   f"{iterations}: The length of the interval is: {round(interval[1] - interval[0],4)}"
        # )

    sol: float = (interval[1] + interval[0]) / 2
    # Check if max iter achieved
    is_good_sol = iterations < max_iter
    # Check if method didn't reach a limit
    if sol < (initial_interval[1] + initial_interval[0]) / 2:
        is_good_sol = is_good_sol and (sol > initial_interval[0] + l)
    else:
        is_good_sol = is_good_sol and (sol < initial_interval[1] - l)
    # print(initial_interval)
    answer: dict = {"value": sol, "converged": is_good_sol, "iterations": iterations}
    return answer
