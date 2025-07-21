import numpy as np
from scipy.linalg import svd, null_space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_orthogonality(A):
    # Compute SVD
    U, S, Vt = svd(A)
    # Determine rank of A
    rank = np.linalg.matrix_rank(A)
    # Row space: First `rank` rows of Vt
    row_space_basis = Vt[:rank]
    # Null space
    null_space_A = null_space(A)
    
    # Plot row space and null space
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot row space vectors
    for i, vec in enumerate(row_space_basis):
        vec = vec / np.linalg.norm(vec)
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='b', linewidth=2, label='Row Space Vector' if i == 0 else None)
        ax.text(vec[0], vec[1], vec[2], f'Row {i+1}', color='b')
    
    # Plot null space vectors
    for i in range(null_space_A.shape[1]):
        nvec = null_space_A[:, i]
        nvec = nvec / np.linalg.norm(nvec)
        ax.quiver(0, 0, 0, nvec[0], nvec[1], nvec[2], color='r', linewidth=2, linestyle='dashed', label='Null Space Vector' if i == 0 else None)
        ax.text(nvec[0], nvec[1], nvec[2], f'Null {i+1}', color='r')
    
    # If row space is a plane (rank 2), plot the plane
    if rank == 2:
        v1 = row_space_basis[0]
        v2 = row_space_basis[1]
        s = np.linspace(-1.5, 1.5, 10)
        t = np.linspace(-1.5, 1.5, 10)
        S, T = np.meshgrid(s, t)
        X = S * v1[0] + T * v2[0]
        Y = S * v1[1] + T * v2[1]
        Z = S * v1[2] + T * v2[2]
        ax.plot_surface(X, Y, Z, alpha=0.2, color='blue')
    
    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title("Orthogonality Between Row Space and Null Space")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Define a 3D matrix
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print("Matrix A:")
    print(A)
    # Visualize orthogonality
    visualize_orthogonality(A)
