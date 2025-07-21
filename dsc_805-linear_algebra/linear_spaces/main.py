import numpy as np

def lu_decomposition(A):
    """Perform LU Decomposition of A = LU"""
    n = len(A)
    #TODO
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_
        # Lower Triangular
        L[i][i] = 1
        for k in range(i+1, n):
            sum_ = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (A[k][i] - sum_) / U[i][i]
    return L, U

def forward_substitution(L, b):
    """Solve Ly = b using forward substitution"""
   #TODO
    n = len(b)
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        sum_ = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - sum_
    return y

def backward_substitution(U, y):
    """Solve Ux = y using backward substitution"""
    #TODO
    n = len(y)
    x = np.zeros_like(y, dtype=float)
    for i in range(n-1, -1, -1):
        sum_ = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_) / U[i][i]
    return x

def solve_system(A, b):
    """Solve the system Ax = b using LU Decomposition"""
    #TODO
    # INSERT_YOUR_CODE
    n = len(A)
    # Initialize L and U as zero matrices
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)
    # Perform LU decomposition
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_ = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_
        # Lower Triangular
        L[i][i] = 1
        for k in range(i+1, n):
            sum_ = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (A[k][i] - sum_) / U[i][i]
    # Forward substitution to solve Ly = b
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        sum_ = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - sum_
    # Backward substitution to solve Ux = y
    x = np.zeros_like(y, dtype=float)
    for i in range(n-1, -1, -1):
        sum_ = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_) / U[i][i]
    return x

# Example
A = np.array([[2, 1, 1], [1, 3, 2], [1, 2, 2]], dtype=float)
b = np.array([4, 5, 6], dtype=float)

# Solve the system
x = solve_system(A, b)

print("Solution to the system is:", x)