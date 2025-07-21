"""
Write a Python function using NumPy to compute the null space of a matrix using the function np.linalg.null_space(). 
Note: Use the scipy.linalg.null_space() for this task   
"""

import numpy as np
from scipy.linalg import null_space

def null_space_matrix(A):
    null_space_basis = null_space(A)
    return null_space_basis 

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(null_space_matrix(A))