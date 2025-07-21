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
A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(f"Matrix A: \n{A}")

# Compute and display the null space of the matrix
null_sp = compute_null_space(A)
print(f"\nComputed Null Space: \n{null_sp}")

# Verification
verification = np.dot(A, null_sp)
print(f"\nVerification (A * Null Space): \n{verification}")