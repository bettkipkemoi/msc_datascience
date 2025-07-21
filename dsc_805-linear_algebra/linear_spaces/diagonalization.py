"""
Write a Python program that computes the eigenvalues and eigenvectors of a given square matrix, 
and reconstructs the matrix using the diagonalization formula
"""

import numpy as np

A = np.array([[4, -5], [2, -3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Diagonal matrix of eigenvalues
Lambda = np.diag(eigenvalues)

# Reconstruct matrix using diagonalization
S = eigenvectors
S_inv = np.linalg.inv(S)
A_reconstructed = S @ Lambda @ S_inv

print("Original matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)
print("\nReconstructed matrix A:")
print(A_reconstructed)

"""
Write a Python program that computes the eigenvalues and eigenvectors of the 
matrix A=[[4, -5], [2, -3]] and reconstructs the matrix using the diagonalization formula A = SΛS
Your program should display: The original matrix A, Eigenvalues, Eigenvectors and the reconstructed matrix 
with the following titles Original matrix A: Eigenvalues: Eigenvectors (columns): Reconstructed matrix A
"""
import numpy as np
# Define the matrix
A = np.array([[4, -5], [2, -3]])
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
# Diagonal matrix of eigenvalues
Lambda = np.diag(eigenvalues)
# Reconstruct matrix using diagonalization
S = eigenvectors
S_inv = np.linalg.inv(S)
A_reconstructed = S @ Lambda @ S_inv
# Print the results
print("Original matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)
print("\nReconstructed matrix A:")
print(A_reconstructed)