import numpy as np
from scipy.linalg import null_space

def compute_null_space(matrix):
    return null_space(matrix)

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(compute_null_space(matrix))
