import numpy as np

def perform_svd(A):
    """
    Perform Singular Value Decomposition on matrix A.
    Returns U, Sigma, Vt such that A = U @ Sigma @ Vt
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Construct the Sigma matrix
    Sigma = np.diag(s)
    return U, Sigma, Vt

# Test matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Perform SVD
U, Sigma, Vt = perform_svd(A)

# Print the results
print("Matrix U:")
print(U)
print("\nMatrix Σ (Sigma):")
print(Sigma)
print("\nMatrix V^T:")
print(Vt)

# Verify reconstruction
A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed Matrix (UΣV^T):")
print(A_reconstructed)

# Check if the reconstruction is close to the original
is_close = np.allclose(A, A_reconstructed)
print("\nReconstruction accurate:", is_close)
