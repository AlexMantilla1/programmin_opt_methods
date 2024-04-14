from typing import Callable, Dict, List
import numpy as np
import numpy.typing as npt


def get_solution_by_golden_section_method(
    epsilon: float,
    initial_interval: List[float],
    obj_fun: Callable[[npt.ArrayLike], float],
    x: npt.ArrayLike,
    direction: npt.ArrayLike,
    max_iter: int = 100,
) -> Dict:
    """This function is just a check for the golden section optimization.
    This function checks if the method converged in the expected iteraitons.
    """
    solution: Dict = golden_section_method(
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
            f"WARNING! golden section algorithm didn't converge in dir: {direction}\n",
            f"Last point: {x}\n",
            f"Solution found: {solution}\n",
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
    initial_interval: List[float],
    obj_fun: Callable[[npt.ArrayLike], float],
    x: npt.ArrayLike,
    direction: npt.ArrayLike,
    max_iter: int = 100,
) -> Dict:
    """
    Golden section method for one-dimensional optimization.

    Parameters:
        l (float): Convergence criterion. Terminate when the length of the interval
                   is less than or equal to l.
        initial_interval (List): Initial interval for the search.
        obj_fun (callable): The objective function to minimize.
        x (array_like): The current point in the optimization process.
        direction (array_like): The search direction.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        Dict: A dictionary containing:
              - 'value': The optimized solution.
              - 'converged': Boolean indicating if the algorithm converged.
              - 'iterations': Number of iterations performed.
    """

    # Setting the first interval to iterate
    interval: List[float] = initial_interval.copy()
    # alpha parameter
    alpha: float = 0.618
    # Calculating lambda and mu for first iteration
    lambda_: float = round(interval[0] + (1 - alpha) * (interval[1] - interval[0]), 4)
    mu: float = round(interval[0] + alpha * (interval[1] - interval[0]), 4)
    iterations: int = 0

    while (interval[1] - interval[0] > l) and (iterations < max_iter):
        # Evaluate the objective function at the evaluating points
        f1 = obj_fun(x + lambda_ * direction)
        f2 = obj_fun(x + mu * direction)

        if f1 <= f2:  # Step 3 from the algorithm
            interval[1] = mu
            # New evaluating points
            mu = lambda_
            lambda_ = round(interval[0] + (1 - alpha) * (interval[1] - interval[0]), 4)
        else:  # Step 2 from the algorithm
            interval[0] = lambda_
            # New evaluating points
            lambda_ = mu
            mu = round(interval[0] + alpha * (interval[1] - interval[0]), 4)

        # Count the iteration
        iterations += 1

    # Once the interval is small enough, set the output
    sol: float = (interval[1] + interval[0]) / 2
    # Check if max iterations achieved
    is_good_sol = iterations < max_iter
    # Check if method converged within the desired precision
    if sol < (initial_interval[1] + initial_interval[0]) / 2:
        is_good_sol = is_good_sol and (sol > initial_interval[0] + l)
    else:
        is_good_sol = is_good_sol and (sol < initial_interval[1] - l)
    # Final solution
    return {
        "value": sol,
        "converged": is_good_sol,
        "iterations": iterations,
    }
