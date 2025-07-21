import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Compute the row space basis using SVD
U, S, Vh = np.linalg.svd(A)
rank = np.linalg.matrix_rank(A)
row_space_basis = Vh[:rank]  # Each row is a basis vector for the row space

# Compute the null space basis (right null space)
def null_space(A, rtol=1e-5):
    u, s, vh = np.linalg.svd(A)
    rank = (s > rtol * s[0]).sum()
    ns = vh[rank:].conj()
    return ns.T

null_basis = null_space(A)
# null_basis is (3, n_null), each column is a null space basis vector

# Pick a vector in the row space (first row of Vh)
row_vec = row_space_basis[0]
# Pick a vector in the null space (first column of null_basis)
null_vec = null_basis[:, 0]

# Normalize for visualization
row_vec = row_vec / np.linalg.norm(row_vec)
null_vec = null_vec / np.linalg.norm(null_vec)

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the row space vector
ax.quiver(0, 0, 0, row_vec[0], row_vec[1], row_vec[2], color='b', label='Row Space Vector', linewidth=2)

# Plot the null space vector
ax.quiver(0, 0, 0, null_vec[0], null_vec[1], null_vec[2], color='r', label='Null Space Vector', linewidth=2)

# Show the plane of the row space (since rank=2, row space is a plane)
if rank == 2:
    # The two basis vectors
    v1 = row_space_basis[0]
    v2 = row_space_basis[1]
    # Create a grid of points in the row space
    s = np.linspace(-1.5, 1.5, 10)
    t = np.linspace(-1.5, 1.5, 10)
    S, T = np.meshgrid(s, t)
    X = S * v1[0] + T * v2[0]
    Y = S * v1[1] + T * v2[1]
    Z = S * v1[2] + T * v2[2]
    ax.plot_surface(X, Y, Z, alpha=0.2, color='blue')

# Set plot limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Row Space and Null Space Orthogonality')

# Show orthogonality visually
# Draw a dotted line showing the dot product is zero
ax.text(null_vec[0], null_vec[1], null_vec[2], 'Null', color='r')
ax.text(row_vec[0], row_vec[1], row_vec[2], 'Row', color='b')

ax.legend()
plt.show()
