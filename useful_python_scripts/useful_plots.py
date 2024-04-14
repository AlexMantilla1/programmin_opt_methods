from typing import Callable, List
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def plot_level_curves_and_points(
    trayectory: npt.ArrayLike, fun: Callable[[npt.ArrayLike], float]
) -> None:
    # Get the x and y values from the trayectory
    points_x = [point[0] for point in trayectory]
    points_y = [point[1] for point in trayectory]
    extra = abs(max(points_x) - min(points_x)) * 0.2
    x_low = min(points_x) - extra
    x_high = max(points_x) + extra
    extra = abs(max(points_y) - min(points_y)) * 0.2
    y_low = min(points_y) - extra
    y_high = max(points_y) + extra

    # Generate grid of points
    x1 = np.linspace(x_low, x_high, 200)
    x2 = np.linspace(y_low, y_high, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = fun(np.array([X1, X2]))

    # Plot level curves
    plt.contour(X1, X2, Z, levels=20)  # Adjust the number of levels as needed
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Add list of points
    plt.scatter(points_x, points_y, color="red", marker="o")

    # Connect points with dashed lines
    plt.plot(points_x, points_y, "r--")

    # Set final details
    plt.title("Level Curves of f(x1, x2) and Opt Trayectory")
    plt.colorbar(label="Function Value")
    plt.grid(True)
    plt.show()


def plot_model_comparison(
    input_signal: List[float],
    output_signal: List[float],
    output_signal_model: List[float],
) -> None:
    n_input = list(range(len(input_signal)))
    n_output = list(range(len(output_signal)))
    plt.figure
    plt.plot(n_input, input_signal, "--s")
    a = plt.plot(n_output, output_signal, "--o")
    plt.plot(n_output, output_signal_model, "--d")
    plt.grid(True)
    plt.axis([-1, 40.0, 3.0, 7.0])
    plt.legend(["Input", "Real Output", "Model Output"])
    plt.show()
