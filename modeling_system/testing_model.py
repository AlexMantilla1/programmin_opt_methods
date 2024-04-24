from typing import Callable, Dict, List
import numpy as np
import numpy.typing as npt
import pandas as pd

# Import my functionalities:
import sys

sys.path.append("../useful_python_scripts/")
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


def evaluate_output_using_model(
    system_model: npt.ArrayLike, input_signal: List[float], initial_condition: float
) -> List[float]:
    # get the order
    order: int = int(len(system_model) / 2)
    # First "order" values are n=-2 and n=-1 (for order = 2) (This assumes steady state with initial condition)
    extended_output: List[float] = order * [initial_condition]
    extended_input: List[float] = (order * [input_signal[0]]) + input_signal
    for n in np.arange(order, len(extended_input)):
        # Create the window that affects the output at n
        interest_window: List[float] = []
        for i in np.arange(order):
            interest_window.append(-extended_output[n - i - 1])
        for i in np.arange(order):
            interest_window.append(extended_input[n - i - 1])

        # The n-th value of the output is
        extended_output.append(np.dot(system_model, interest_window))

    return extended_output[order:]


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


def test_model(model_to_test: npt.ArrayLike) -> None:
    # Reading data:
    input_signal, output_signal = get_all_data()

    # Evaluate output with model
    output_signal_model = evaluate_output_using_model(
        model_to_test, input_signal, output_signal[0]
    )

    # Calculate objective function (cuadratic mean error)
    function_value = calculate_cuadratic_mean_error(output_signal_model, output_signal)
    print(f"f achieved: {function_value}")

    # plot the model to compare
    plot_model_comparison(input_signal, output_signal, output_signal_model)


def main() -> None:
    # Enter here the model to test
    # ̂y = -(0.73821)y[k-2] - (-1.68868)y[k-1] + (0.04415)u[k-2] + (0.00383)u[k-1]
    model_to_test = [-1.68868, 0.73821, 0.00383, 0.04415]
    # This function tests the model
    test_model(np.array(model_to_test))


if __name__ == "__main__":
    main()
