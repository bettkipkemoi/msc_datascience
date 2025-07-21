"""
You are given the matrix
[[1 2 3]
 [4 5 6]
 [7 8 9]]
Perform SVD decomposition on , i.e., compute such that:
Write a Python function rank_k_approximation(A, k) that computes the best rank approximation of for given . 
Compute and print:
The original matrix .
Its rank-1 and rank-2 approximations.
The Frobenius norm of the difference between the original matrix and each approximation.
The program should display:
Original Matrix A:
Rank-1 Approximation:
Frobenius Norm of Difference: 
Rank-2 Approximation:
Frobenius Norm of Difference: 
"""
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
              [7, 8, 9]])

# Desired rank
k1 = 1
k2 = 2

# Compute rank 1 approximation
A_1 = optimal_low_rank_approximation(A, k1)
# Rank-2 approximation
A_2 = optimal_low_rank_approximation(A, k2)

# Function to compute the Frobenius norm of the difference
def frobenius_norm(A, A_k):
    return np.linalg.norm(A - A_k, 'fro')
# Compute the Frobenius norm of the difference
error = frobenius_norm(A, A_1)
error2 = frobenius_norm(A, A_2)

print("Original Matrix A:")
print(A)
print("\nRank-1 Approximation:")
print(A_1)
print("Frobenius Norm of Difference:", round(error, 2))
print("\nRank-2 Approximation:")
print(np.around(A_2,1))
#print(np.around(A, decimals=2))
print("Frobenius Norm of Difference:", "{:.2f}".format(error2))
