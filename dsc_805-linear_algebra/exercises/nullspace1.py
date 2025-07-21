# import numpy
import numpy as np
from scipy.linalg import null_space

# Function to compute the null space of the matrix
def compute_null_space(matrix):
    """Returns the null space of the matrix."""
    # Use scipy's null_space function to calculate the null space
    null_sp = null_space(matrix)
   
    return null_sp

#example matrix
matrix = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
# Compute and display the null space of the matrix
print(compute_null_space(matrix))

import numpy as np

def compute_null_space(A, tol=1e-10):
    """
    Compute the null space of a matrix A using SVD.
    
    Parameters:
    A : numpy.ndarray
        Input matrix
    tol : float
        Tolerance for considering singular values as zero (default: 1e-10)
    
    Returns:
    null_space : numpy.ndarray
        Orthonormal basis for the null space of A
    """
    # Compute SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    
    # Identify zero singular values (within tolerance)
    rank = np.sum(s > tol)
    
    # Extract columns of Vh.T (right singular vectors) corresponding to zero singular values
    null_space = Vh[rank:].T
    
    return null_space

# Example matrix
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Compute null space
null_space = compute_null_space(matrix)

# Print results
print("Matrix A:")
print(matrix)
print("\nNull space basis:")
print(null_space)
print("\nVerification (A @ null_space should be close to zero):")
print(matrix @ null_space)