
from typing import Callable
import numpy as np
import numpy.typing as npt 

def orthogonalize_vectors(input_vectors: list[dict]) -> npt.ArrayLike:
    a_list: list[npt.ArrayLike] = []
    non_used_vectors: list[dict] = input_vectors.copy()
    for vector in input_vectors:
        if vector["opt_sol"] == 0:
            a_list.append(vector["direction"])
        else:
            temp_vector: npt.ArrayLike = np.array([0,0])
            for non_used_vec in non_used_vectors:
                temp_vector -= non_used_vec["opt_sol"]*non_used_vec["direction"]
            a_list.append(temp_vector)
        
        non_used_vectors.remove(vector)
        
        
    b = []
    used_as = []
    for a in a_list:
        temp_vector: npt.ArrayLike = a
        if np.array_equal(a,a_list[0]):
            b.append(temp_vector)
        else:
            for 
            
           