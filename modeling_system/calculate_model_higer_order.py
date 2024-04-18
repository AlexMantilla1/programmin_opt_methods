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
    return [input_signal[000:200], output_signal[000:200]]
    # return [input_signal, output_signal]


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
    # output_signal_model = evaluate_output_using_model_v2(
    #    test_model, input_signal, output_signal
    # )
    # Return the cuadratic mean error
    return calculate_cuadratic_mean_error(output_signal_model, output_signal)


def cal_det_coef(real_output: List[float], model_output: List[float]) -> float:
    """
    Calculate the determination coefficient (R^2) to assess the similarity
    or correlation between two sets of data.

    Parameters:
        real_output (List[float]): The observed or real values.
        model_output (List[float]): The predicted or model values.

    Returns:
        float: The determination coefficient (R^2).
    """
    # Convert input lists to numpy arrays
    real_output = np.array(real_output)
    model_output = np.array(model_output)

    # Compute the mean of real_output and model_output
    mean_real = np.mean(real_output)

    # Compute the total sum of squares (SST)
    sst = np.sum((model_output - mean_real) ** 2)

    # Compute the residual sum of squares (SSE)
    sse = np.sum((real_output - mean_real) ** 2)

    # Compute R^2
    r_squared = sst / sse

    return r_squared


def test_model(model_to_test: npt.ArrayLike) -> None:
    # Reading data:
    input_signal, output_signal = get_all_data()

    output_signal_model = evaluate_output_using_model(
        model_to_test, input_signal, output_signal[0]
    )
    # output_signal_model = evaluate_output_using_model_v2(
    #    model_to_test, input_signal, output_signal
    # )

    # Calculate objective function (cuadratic mean error)
    function_value = calculate_cuadratic_mean_error(output_signal_model, output_signal)
    print(f"f achieved: {function_value}")

    # Calculate determination coeficient.
    R_2 = cal_det_coef(output_signal, output_signal_model)
    print(f"R_2 = {R_2}")

    # plot the model to compare
    plot_model_comparison(input_signal, output_signal, output_signal_model)


def main() -> None:
    # Reading data:
    # input_signal, output_signal = get_data()

    # Define an arbitrary initial point for training
    initial_model: npt.ArrayLike = np.array(
        [-1.50679337, 0.57994944, -0.04300483, 0.11451339]
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
    # print(f"Solution found: {system_model} (iters required: {solution["iterations"]})")
    # Test the model
    test_model(system_model)


if __name__ == "__main__":
    main()
    # model_to_test: npt.ArrayLike = np.array(
    #    [-1.50323926, 0.57722442, -0.04175521, 0.11251785]
    # )
    # test_model(np.array([-0.39587865, -0.49787017, 0.00302282, 0.10024205]))
