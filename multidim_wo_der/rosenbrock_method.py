from typing import Callable, List, Dict
import numpy as np
import numpy.typing as npt
from golden_section_method_Rn import get_solution_by_golden_section_method
from useful_plots import plot_level_curves_and_points

def calculate_vectors_by_gram_schmidt(input_vectors: List[npt.ArrayLike]) -> List[npt.ArrayLike]:
    """
    Calculate orthogonal basis using the Gram-Schmidt method.

    Parameters:
    - input_vectors (List[npt.ArrayLike]): A list of input vectors forming the original basis.

    Returns:
    - List[npt.ArrayLike]: A list of vectors forming an orthogonal basis.
    """

    # Get the number of input vectors
    num_vectors = len(input_vectors)

    # Initialize list to store orthogonal basis vectors
    b_list: List[npt.ArrayLike] = []

    # Iterate over each input vector
    for j in np.arange(num_vectors):
        temp_vec: npt.ArrayLike = input_vectors[j]

        # For the first vector, add it directly to the orthogonal basis
        if j == 0:
            b_list.append(temp_vec)
        else:
            # For subsequent vectors, subtract projections onto previous orthogonal basis vectors
            for i in np.arange(0, j):
                temp_vec = temp_vec - (np.dot(input_vectors[j], b_list[i]) / np.dot(b_list[i], b_list[i])) * b_list[i]

            # Add the resulting orthogonal vector to the basis
            b_list.append(temp_vec)

    return b_list

# in the book input_vectors are called "d"
""" Algorithm for this funct: 
    1. Calc a's
    2. Calc b's 
    3. Calc new d's  (output_vectors)
    """
def orthogonalize_and_orthonormalize_vectors(input_vectors: List[npt.ArrayLike], opt_sols: List[float]) -> List[npt.ArrayLike]:
    """
    Orthogonalize and orthonormalize a set of input vectors.

    Parameters:
    - input_vectors (List[npt.ArrayLike]): A list of input vectors forming the original basis.
    - opt_sols (List[float]): A list of optimization coefficients obtained from a method like linear regression.

    Returns:
    - List[npt.ArrayLike]: A list of vectors forming an orthogonal and orthonormal basis.
    """

    # Get the number of input vectors
    num_vectors = len(input_vectors)
    # 1. Calculate the 'a' vectors
    # Initialize list to store 'a' vectors
    a_list: List[npt.ArrayLike] = []   

    # Calculate each 'a' vector
    for j in np.arange(num_vectors):
        # If lambda_j = 0, set the input vector as 'a'
        if opt_sols[j] == 0.0:
            a_list.append(input_vectors[j])
        else: 
            # Otherwise, calculate the linear combination of input vectors based on optimization coefficients
            # Initialize accumulator vector
            temp_vec: npt.ArrayLike = np.array( [0.0, 0.0] )
            # Accumulate through the input vectors
            for i in np.arange(j,num_vectors):
                temp_vec += opt_sols[i]*input_vectors[i]
            # Append new a vector
            a_list.append(temp_vec)

    # At this point a's are calculated.

    # 2. Calculate the 'b' vectors using Gram-Schmidt method
    b_list: npt.ArrayLike = calculate_vectors_by_gram_schmidt(a_list)

    # 3. Orthonormalize the calculated directions (b's)
    output_vectors = [(1/np.linalg.norm(vec))*vec if np.linalg.norm(vec) > 0 else vec for vec in a_list]
    
    # Return the calculated vectors
    return output_vectors
            

def rosenbrock_method(
    epsilon: float,
    initial_point: npt.ArrayLike,
    obj_function: Callable[[npt.ArrayLike], float],
    num_dim: int,
    max_iter: int = 500,
    max_step: float = 10.0,
) -> Dict:
    """
    Rosenbrock Method

    Parameters:
    - epsilon (float): The termination scalar to define the threshold of how close is close enough for the solution.
    - initial_point (npt.ArrayLike): The initial point for the algorithm.
    - obj_function (Callable[[npt.ArrayLike], float]): The function we're minimizing.
    - num_dim (int): The number of dimensions of the obj_function (in f: R^n -> R, the value of n).
    - max_iter (int): Max number of iterations allowed to converge. Default is 500.
    - max_step (float): Every iteration will have a step in every dimension, this defines the maximum step. Default is 10.0.

    Returns:
    - Dict: A dictionary containing the result of the optimization process, including 'value' (the optimized point), 'converged' (boolean indicating convergence), 'iterations' (number of iterations performed), and 'trayectory' (list of all points explored during the optimization).
    """
    
    # Set up
    iterations: int = 0
    last_point: npt.ArrayLike = initial_point.copy()
    new_point: npt.ArrayLike = last_point.copy()
    directions: List[npt.ArrayLike] = list(np.eye(num_dim))
    all_points_trayectory = [initial_point]
    
    # Main loop
    while iterations < max_iter:
        
        iterations += 1
        # 1. Exploratory search in the specified direction
        #   Iterate for every dimention to find a new opt point. 
        opt_solutions: List[float] = []
        for dir_index in np.arange(num_dim): 
            # Perform golden section search along this direction
            solution: Dict = get_solution_by_golden_section_method(
                epsilon / 4,
                [-max_step, max_step],
                obj_function,
                new_point,
                directions[dir_index],
            )
            # Check if solution was found
            if not solution["converged"]:
                solution["trayectory"] = all_points_trayectory
                return solution
            # Move the point the dir_index dimention by the opt solution found 
            new_point += solution["value"]*directions[dir_index]
            # Save the optimal solution found
            opt_solutions.append(solution["value"])
            # Add the point to the trayectory
            all_points_trayectory.append(new_point.copy())
            
        
        # End the algorithm if the magnitude of the difference is lower that epsilon
        magnitude: float = np.linalg.norm(new_point - last_point)
        if magnitude < epsilon:
            break 

        # Update the last point for next iteration
        last_point = new_point.copy() 
        # 2. Calculate a new direction from the new point (Using Gram-Schmith ) 
        directions = orthogonalize_and_orthonormalize_vectors(directions,opt_solutions)
        
    # Check if max iterations achieved
    is_good_sol: bool = iterations < max_iter
    return {
        "value": new_point,
        "converged": is_good_sol,
        "iterations": iterations,
        "trayectory": all_points_trayectory,
    } 

def function_R2(x: npt.ArrayLike) -> float:
    """
    Example function to test.
    For f: R2 -> R
    """
    x1 = x[0]
    x2 = x[1]
    return (x1 - 2) ** 4 + (x1 - (2 * x2)) ** 2


def function_R2_b(x: npt.ArrayLike) -> float:
    """
    Example function to test.
    For f: R2 -> R
    """
    x1 = x[0]
    x2 = x[1]
    return (x1 - 2) ** 2 + (x2 - 4) ** 2



def main() -> None:
    # Define a termination scalar, to define the threshold of how much is close enough for the solution
    epsilon: float = 0.001
    # Define an arbitrary point
    # initial_point: npt.ArrayLike = np.array([-2.1, 7.0])
    initial_point: npt.ArrayLike = np.array([0.0, 3.0])
    # The function to minize:
    fun: Callable[[npt.ArrayLike], float] = function_R2
    # Call the function for cyclic coordinate algorithm using golden section method.
    solution: Dict = rosenbrock_method(epsilon, initial_point, fun, 2)
    # Just print solution
    print(f"The solution is: {solution["value"]}")
    # Print the trayectory len
    print(f"Points in trayectory: {len(solution["trayectory"])}")

    if solution["converged"]:
        # Make the level curves plot
        plot_level_curves_and_points(solution["trayectory"],fun)
    
if __name__ == "__main__":
    main()
    
    
