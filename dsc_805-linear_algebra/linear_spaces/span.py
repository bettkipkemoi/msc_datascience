import numpy as np

# Define the vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([4, 5, 6])

# Stack the vectors into a matrix
matrix = np.column_stack((v1, v2, v3))

# Perform row-reduction (compute rank and find a basis)
_, _, vh = np.linalg.svd(matrix)

# Tolerance for numerical rank determination
tol = 1e-10
rank = np.sum(np.abs(vh) > tol, axis=0)

# Identify the basis vectors (linearly independent columns)
basis_indices = np.where(rank)[0]
basis_vectors = matrix[:, basis_indices]
print(basis_vectors)