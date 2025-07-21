#import numpy
import numpy as np

def optimal_low_rank_approximation(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    A_k = U_k @ S_k @ Vt_k
    return A_k

# Original matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# Desired rank
k = 2

# Compute rank-k approximation
A_k = optimal_low_rank_approximation(A, k)

# Compute Frobenius norms
fro_A = np.linalg.norm(A, 'fro')
fro_Ak = np.linalg.norm(A, 'fro')

# Compute Spectral norms (2-norm)
spec_A = np.linalg.norm(A, 2)
spec_Ak = np.linalg.norm(A, 2)

print("Frobenius norm of A:",fro_A)
print("Frobenius norm of A_k:",fro_Ak)
print("Spectral norm of A:",spec_A)
print("Spectral norm of A_k:",spec_Ak)