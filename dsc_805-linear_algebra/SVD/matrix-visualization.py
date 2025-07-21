"""
Goal: Visualize the singular values and understand their significance in matrix approximation.

Instructions:

Write Python code to:
Perform SVD on a matrix A using numpy.linalg.svd().
Plot the singular values from the diagonal matrix Σ using matplotlib.
Analyze the contribution of each singular value to the overall matrix by:
Truncating smaller singular values.
Reconstructing the matrix using only the top k singular values, where k is a smaller number than the rank of A.
Test this on the matrix:
 A = [[3,2,2],[2,3,-2]]
and plot the matrix approximations for different values of k.

Expected Outcome: A plot showing the singular values and matrix approximations for different values of k, illustrating the role of singular values in capturing the essential structure of a matrix.
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the matrix
A = np.array([[3, 2, 2],
              [2, 3, -2]])

# Perform SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
Sigma = np.diag(s)

# Plot the singular values
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(np.arange(1, len(s)+1), s, 'o-', linewidth=2)
plt.title('Singular Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Function to reconstruct A using top k singular values
def reconstruct_matrix(U, s, Vt, k):
    Sigma_k = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(Sigma_k, s[:k])
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

# Plot matrix approximations for k=1 and k=2
ks = [1, 2]
for idx, k in enumerate(ks, start=2):
    A_k = reconstruct_matrix(U, s, Vt, k)
    plt.subplot(1, 3, idx)
    plt.imshow(A_k, cmap='viridis', aspect='auto')
    plt.title(f'Approximation with k={k}')
    plt.colorbar()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))

plt.suptitle('SVD: Singular Values and Matrix Approximations')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Print analysis
print("Original Matrix A:")
print(A)
print("\nSingular Values:")
print(s)
for k in ks:
    A_k = reconstruct_matrix(U, s, Vt, k)
    print(f"\nReconstructed Matrix with k={k}:")
    print(A_k)
    error = np.linalg.norm(A - A_k)
    print(f"Reconstruction Error (Frobenius norm): {error:.4f}")
