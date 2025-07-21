"""
Goal: Use SVD to solve a system of linear equations.

Instructions:

Consider the linear system Ax = b, where A is a 3*3 matrix and b is a vector:
A = [1, 2, 3; 4, 5, 6; 7, 8, 9]
b = [3, 3, 4] 
 
Solve for x using SVD. First, decompose A into U, Σ, and V^T, and then use the decomposition to find the solution.
Compare your solution to that obtained using numpy.linalg.solve().
Expected Outcome: A solution to the linear system using SVD and verification by comparing the results with other standard methods.
"""
