import numpy as np

# Define the vectors
v1 = np.array([1, 2])
v2 = np.array([2, 4])
v3 = np.array([3, 6])

# Stack the vectors into a matrix
matrix = np.column_stack((v1, v2, v3))

# Perform row-reduction (compute the rank)
rank = np.linalg.matrix_rank(matrix)
# Check if the vectors are linearly dependent
if rank < matrix.shape[1]:
    print("The vectors are linearly dependent.")
    # Solve for the linear dependence relationship
    _, _, vh = np.linalg.svd(matrix)
    linear_combination = vh[-1]  # The null space vector
    print(f"{linear_combination[0]} * v1 + {linear_combination[1]} * v2 + {linear_combination[2]} * v3 = 0")
else:
    print("The vectors are linearly independent.")