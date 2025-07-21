import numpy as np
import matplotlib.pyplot as plt
# Function to perform spectral decomposition
def spectral_decomposition(A):
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Reconstruct the matrix from the eigenvalues and eigenvectors
    Lambda = np.diag(eigenvalues) # Diagonal matrix of eigenvalues
    Q = eigenvectors # Matrix of eigenvectors
    # Verify that A = Q * Lambda * Q^-1
    A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)
    return eigenvalues, eigenvectors, A_reconstructed
# Example matrix
A = np.array([[4, 2],
    [1, 3]])
# Perform spectral decomposition
eigenvalues, eigenvectors, A_reconstructed = spectral_decomposition(A)
# Print results
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
print("\nReconstructed Matrix (A):")
print(A_reconstructed)
# Visualize the matrix, eigenvalues, and eigenvectors
def visualize_spectral_decomposition(A, eigenvalues, eigenvectors):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Visualize the matrix A
    ax[0].imshow(A, cmap='coolwarm', interpolation='nearest')
    ax[0].set_title('Matrix A')
    ax[0].set_xticks(range(A.shape[0]))
    ax[0].set_yticks(range(A.shape[1]))
    # Visualize eigenvectors and eigenvalues
    ax[1].quiver(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], angles='xy')
    ax[1].quiver(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], angles='xy')
    ax[1].set_xlim([-1.5, 1.5])
    ax[1].set_ylim([-1.5, 1.5])
    ax[1].grid(True)
    ax[1].set_title('Eigenvectors and Eigenvalues')
    ax[1].legend()
    plt.show()
# Visualize the spectral decomposition results
visualize_spectral_decomposition(A, eigenvalues, eigenvectors)



"""
Given the following vector [[4, 1, 2, 3], [1, 4, 3, 2], [2, 3, 4, 1], [3, 2, 1, 4]]. compute its eigenvalues and eigenvectors using Python. Additionally, verify that the computed eigenvectors are indeed correct.
Display the eigenvectors, the eigenvalues and  the verification results.
Your output should have headers as follows
Eigenvalues:
Eigenvectors:
Verification Results:
"""

import numpy as np
# function to perform spectral decomposition
def spectral_decomposition(A):
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Reconstruct the matrix from the eigenvalues and eigenvectors
    Lambda = np.diag(eigenvalues)  # Diagonal matrix of eigenvalues
    Q = eigenvectors  # Matrix of eigenvectors
    # Verify that A = Q * Lambda * Q^-1
    A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)
    return eigenvalues, eigenvectors, A_reconstructed
# Example matrix
A = np.array([[4, 1, 2, 3],
              [1, 4, 3, 2],
              [2, 3, 4, 1],
              [3, 2, 1, 4]])
# Perform spectral decomposition
eigenvalues, eigenvectors, A_reconstructed = spectral_decomposition(A)
# Print results
print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
#verification of the results
# Verify that A @ v = lambda * v for each eigenpair
verification = []
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    Av = A @ v
    lv = eigenvalues[i] * v
    # Use np.allclose to check if Av and lv are approximately equal
    verification.append(np.allclose(Av, lv))

print("\nVerification Results:")
print(verification)

