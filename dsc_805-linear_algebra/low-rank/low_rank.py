#import numpy
import numpy as np

A = np.array([[4, 1, 2, 3], [1, 4, 3, 2], [2, 3, 4, 1], [3, 2, 1, 4]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
#verification
print("Verification Results:")
verification_results = []
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    Av = A @ v
    lv = eigenvalues[i] * v
    is_close = np.allclose(Av, lv)
    verification_results.append(bool(is_close))
print(verification_results)
    