"""
Implement the Gram-Schmidt orthogonalization process in Python. 
Given a set of linearly independent vectors, return an orthonormal set of vectors.
"""

import numpy as np

def gram_schmidt(V):
    Q = []
    for v in V:
        # Orthogonalize v against the vectors in Q
        v_orth = v - sum(np.dot(q, v) * q for q in Q)
        # Normalize the orthogonalized vector
        q = v_orth / np.linalg.norm(v_orth)
        Q.append(q)
    return np.array(Q)
# Input matrix V
V = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])

# Apply Gram-Schmidt process
Q = gram_schmidt(V)

# Output the result
print("Orthonormal basis:")
print(Q)

#approach 2
import numpy as np
def gram_schmidt(vectors):
    """
    Gram-Schmidt orthogonalization process.
    Input: A list or array of numpy vectors.
    Output: An orthonormal basis as a list of numpy vectors.
    """
    # Initialize an empty list to hold the orthonormal basis
    orthonormal_basis = []
    for v in vectors:
        # Start with the current vector
        u = np.copy(v)
        # Subtract the projection of v onto all previously computed orthonormal vectors
        for q in orthonormal_basis:
            proj = np.dot(v, q) * q
            u = u - proj
        # Normalize the resulting vector
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-10:  # Avoid dividing by zero
            q = u / u_norm  # Normalized orthogonal vector
            orthonormal_basis.append(q)
    return orthonormal_basis

# Example usage:
# Input set of vectors (columns of matrix)
V = np.array([
    [1, 1, 1],
    [1, 0, 2],
    [1, 1, 0]
], dtype=float)

# Perform Gram-Schmidt orthogonalization
orthonormal_basis = gram_schmidt(V.T)

# Output the orthonormal basis
print("Orthonormal Basis:")
for q in orthonormal_basis:
    print(q)


"""
Implement the Gram-Schmidt orthogonalization process in Python. Given a set of linearly independent vectors[[1, 1, 0], [1, 0, 1], [0, 1, 1]] , return an orthonormal set of vectors.
Display your output with the heading 
Orthonormal basis:
"""
import numpy as np
def gram_schmidt(V):
    Q = []
    for v in V:
        # Orthogonalize v against the vectors in Q
        v_orth = v - sum(np.dot(q, v) * q for q in Q)
        # Normalize the orthogonalized vector
        q = v_orth / np.linalg.norm(v_orth)
        Q.append(q)
    return np.array(Q)
# Input matrix V
V = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])

# Apply Gram-Schmidt process
Q = gram_schmidt(V)

# Output the result
print("Orthonormal basis:")
print(Q)
