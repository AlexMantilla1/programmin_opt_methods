from typing import Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd

# Import my functionalities:
import sys

sys.path.append("../useful_python_scripts/")
from multidim_opt_methods import (
    davidson_fletcer_powell,
    newton_method,
    steepest_decent_method,
    levenberg_marquadt,
)
from useful_plots import plot_model_comparison


def get_all_data() -> List[List]:
    # Read the CSV file
    # print("Reading data...")
    df = pd.read_csv("data.csv")

    # Extract each column as a vector
    input_signal = df["input"].tolist()
    output_signal = df["output"].tolist()

    # Return the data
    return [input_signal, output_signal]

    # return [input_signal, output_signal]


def get_data() -> List[List]:
    # Read the CSV file
    # print("Reading data...")
    df = pd.read_csv("data.csv")

    # Extract each column as a vector
    input_signal = df["input"].tolist()
    output_signal = df["output"].tolist()

    # Return the data
    # return [input_signal[000:200], output_signal[000:200]]
    return [input_signal, output_signal]


def evaluate_output_using_model(
    system_model: npt.ArrayLike, input_signal: List[float], initial_condition: float
) -> List[float]:
    # get the order
    order: int = int(len(system_model) / 2)

    # First two values are n=-2 and n=-1 (This assumes steady state with initial condition)
    extended_output: List[float] = order * [initial_condition]
    extended_input: List[float] = (order * [input_signal[0]]) + input_signal
    for n in np.arange(order, len(extended_input)):

        """# The values that affect the output:
        interest_window: npt.ArrayLike = np.array(
            [
                -extended_output[n - 1],
                -extended_output[n - 2],
                extended_input[n - 1],
                extended_input[n - 2],
            ]
        )"""
        # Create the window that affects the output at n
        interest_window: List[float] = []
        for i in np.arange(order):
            interest_window.append(-extended_output[n - i - 1])
        for i in np.arange(order):
            interest_window.append(extended_input[n - i - 1])

        # The n-th value of the output is
        extended_output.append(np.dot(system_model, interest_window))

    return extended_output[order:]


def evaluate_output_using_model_v2(
    system_model: npt.ArrayLike, input_signal: List[float], output_signal: List[float]
) -> List[float]:
    # get the order
    order: int = int(len(system_model) / 2)

    # First two values are n=-2 and n=-1 (This assumes steady state with initial condition)
    model_output: List[float] = output_signal[0:order].copy()
    for n in np.arange(order, len(input_signal)):
        # Create the window that affects the output at n
        interest_window: List[float] = []
        for i in np.arange(order):
            interest_window.append(-output_signal[n - i - 1])
        for i in np.arange(order):
            interest_window.append(input_signal[n - i - 1])

        # The n-th value of the output is
        model_output.append(np.dot(system_model, interest_window))

    return model_output


def calculate_cuadratic_mean_error(
    output_signal_model: List[float], real_output: List[float]
) -> float:
    cuadratic_mean_error: float = 0.0
    N: int = len(real_output)
    # Acumulate the Cuadratic mean error
    for n in np.arange(N):
        cuadratic_mean_error = (
            cuadratic_mean_error + (real_output[n] - output_signal_model[n]) ** 2
        )
    # Return the cuadratic mean error
    return cuadratic_mean_error / N


# Cuadratic mean error as obj_function of the model
def objective_function(test_model: npt.ArrayLike) -> float:
    # Get data to compare
    input_signal, output_signal = get_data()
    # Evaluate output with model
    output_signal_model = evaluate_output_using_model(
        test_model, input_signal, output_signal[0]
    )
    # Return the cuadratic mean error
    return calculate_cuadratic_mean_error(output_signal_model, output_signal)


def test_model(model_to_test: npt.ArrayLike) -> None:
    # Reading data:
    input_signal, output_signal = get_all_data()

    output_signal_model = evaluate_output_using_model(
        model_to_test, input_signal, output_signal[0]
    )

    # Calculate objective function (cuadratic mean error)
    function_value = calculate_cuadratic_mean_error(output_signal_model, output_signal)
    print(f"f achieved: {function_value}")

    # plot the model to compare
    plot_model_comparison(input_signal, output_signal, output_signal_model)


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
