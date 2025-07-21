import numpy as np
from scipy.linalg import null_space
# Function to compute and display the column space
def column_space(matrix):
    """Returns the column space of the matrix (set of linearly independent columns)."""
    # Perform QR decomposition to get the orthogonal basis for the column space
    Q, R = np.linalg.qr(matrix)
    # Extract the number of independent columns by checking the rank of R
    rank = np.linalg.matrix_rank(matrix)
    # The first 'rank' columns of Q form the basis for the column space
    col_space = Q[:, :rank]
    print(f"Column Space (Basis):\n{col_space}\n")
    return col_space
# Function to compute the null space of the matrix
def compute_null_space(matrix):
    """Returns the null space of the matrix."""
    # Use scipy's null_space function to calculate the null space
    null_sp = null_space(matrix)
    if null_sp.size == 0:
        print("The null space is trivial (only contains the zero vector).")
    else:
        print(f"Null Space (Basis):\n{null_sp}\n")
    return null_sp
# Function to compute the rank of the matrix
def compute_rank(matrix):

        """Returns the rank of the matrix."""
        rank = np.linalg.matrix_rank(matrix)
        print(f"Rank of the matrix: {rank}\n")
        return rank
# Example matrix
matrix = np.array([
    [2, 4, 1],
    [0, 1, -1],
    [-2, -4, 5]
])
print("Matrix:")
print(matrix)
# Compute and display the rank of the matrix
rank = compute_rank(matrix)
# Compute and display the column space of the matrix
col_space = column_space(matrix)
# Compute and display the null space of the matrix
print(compute_null_space(matrix))