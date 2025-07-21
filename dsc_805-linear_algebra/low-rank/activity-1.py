"""
Write python code performs Singular Value Decomposition (SVD) and computes the best rank-k approximation of the matrix A=[[1, 2, 3], [4, 5, 6], [7, 8, 9]].
Your output should have the title: Best rank-k approximation of A:
"""
#import numpy
import numpy as np
def optimal_low_rank_approximation(A, k):
    """
    Compute the optimal low-rank approximation of matrix A of rank k.
    Parameters:
    A: The original matrix
    k: Desired rank for the approximation
    Returns:
    A_k: The low-rank approximation of A
    """
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Keep only the top k singular values and corresponding singular vectors
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # Compute the Low-rank approximation
    A_k = U_k @ S_k @ Vt_k
    return A_k

# Example matrix
A = np.array(([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]), dtype=float)
# Desired rank for the approximation
k = 2
# Compute the optimal Low-rank approximation
A_k = optimal_low_rank_approximation(A, k)

# Print the Low-rank approximation
print("Best rank-k approximation of A:")
print("", A_k)