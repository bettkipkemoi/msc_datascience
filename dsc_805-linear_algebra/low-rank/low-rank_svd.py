import numpy as np
import matplotlib.pyplot as plt
# Function to compute the optimal Low-rank matrix approximation
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
    
# Function to compute the Frobenius norm of the difference
def frobenius_norm(A, A_k):
    return np.linalg.norm(A - A_k, 'fro')
# Example matrix
A = np.array([[4, 2, 3, 1],
    [1, 3, 2, 4],
    [2, 4, 1, 5],
    [6, 1, 0, 3]], dtype=float)
# Desired rank for the approximation
k = 2
# Compute the optimal Low-rank approximation
A_k = optimal_low_rank_approximation(A, k)

# Print the original matrix and the Low-rank approximation
print("Original Matrix (A):")
print(A)
print("\nLow-Rank Approximation (A_k) with rank", k, ":")
print(A_k)

# Compute the Frobenius norm of the difference
error = frobenius_norm(A, A_k)
print("\nFrobenius Norm of the difference between A and A_k:", error)

# Visualizing the original matrix and the Low-rank approximation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original Matrix (A)")
plt.imshow(A, cmap='viridis')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title(f"Low-Rank Approximation (A_k) with rank {k}")
plt.imshow(A_k, cmap='viridis')
plt.colorbar()
plt.show()