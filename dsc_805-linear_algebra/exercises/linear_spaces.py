import numpy as np
from scipy.linalg import null_space
from scipy.linalg import svd

# Define matrix A
A = np.array([[1, 2, 0],
              [2, 4, 0],
              [0, 0, 3]])

m, n = A.shape

# Compute rank
rank = np.linalg.matrix_rank(A)

# Column space (basis): columns of U corresponding to nonzero singular values
U, s, Vh = svd(A)
col_space_basis = U[:, :rank]

# Row space (basis): rows of Vh corresponding to nonzero singular values
row_space_basis = Vh[:rank, :]

# Null space (basis)
nullspace_basis = null_space(A)
nullity_computed = nullspace_basis.shape[1]

# Left null space (basis): null space of A.T
left_nullspace_basis = null_space(A.T)

# Theoretical nullity
nullity_theoretical = n - rank

# Rank-Nullity Theorem Verification
rank_nullity_verified = (rank + nullity_computed == n)

print(f"Column Space (basis): {col_space_basis}")
print(f"\nRow Space (basis): \n{row_space_basis}")
print(f"\nNull Space (basis): \n{nullspace_basis}")
print(f"\nLeft Null Space (basis): \n{left_nullspace_basis}")
print("\nRank-Nullity Theorem Verification:")
print(f"  Rank (computed): {rank}")
print(f"  Nullity (theoretical): {nullity_theoretical}")
print(f"  Nullity (computed): {nullity_computed}")
print(f"  Rank-Nullity Verified: {rank_nullity_verified}")