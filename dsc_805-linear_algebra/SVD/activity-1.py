"""
In this task, you will visualize the action of a matrix 
A=[[1, 0], [0, 1], [1, 1]] on various vectors using Python’s Matplotlib. Specifically, you will observe the transformations that lead to the Singular Value Decomposition (SVD) of matrix 
A. The goal is to understand how the matrix affects vectors in terms of scaling, rotating, and projecting onto new axes. You will also observe how the Singular Value Decomposition (SVD) can decompose the matrix into three components, representing rotation, scaling, and rotation again.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(ax, vectors, color, label=None):
    """
    Plot a set of vectors on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis on which to plot the vectors.
        vectors (numpy.ndarray): The vectors to plot (each column is a vector).
        color (str): The color of the vectors.
        label (str, optional): The label for the vectors.
    """
    # TODO: Implement the vector plotting logic here.
    origin = np.zeros((2, vectors.shape[1]))
    for i in range(vectors.shape[1]):
        ax.quiver(
            origin[0, i], origin[1, i],
            vectors[0, i], vectors[1, i],
            angles='xy', scale_units='xy', scale=1,
            color=color, alpha=0.8,
            label=label if label and i == 0 else None
        )

def visualize_matrix_action(A):
    """
    Visualize the action of matrix A on vectors and the SVD decomposition.

    Parameters:
        A (numpy.ndarray): The matrix to analyze.
    """
    # Define a set of 2D vectors
    vectors = np.array([[1, 0], [0, 1], [1, 1]]).T

    # TODO: Compute the SVD of matrix A.
    A = np.array([[1, 0], [0, 1], [1, 1]])
    U, S, Vt = np.linalg.svd(A)
    # TODO: Create the plot and visualize original and transformed vectors.
    v = np.array([1, 0])
    u = A @ v
    plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='b', label='Av')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.legend()
    plt.show()

    # TODO: Plot the singular vectors and their effect.
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(1, len(S)+1), S, 'o-', linewidth=2)
    plt.title('Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

    # Function to reconstruct A using top k singular values
    def reconstruct_matrix(U, s, Vt, k):
        Sigma_k = np.zeros((U.shape[1], Vt.shape[0]))
        np.fill_diagonal(Sigma_k, s[:k])
        return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    # Plot matrix approximations for k=1 and k=2
    ks = [1, 2]
    for idx, k in enumerate(ks, start=2):
        A_k = reconstruct_matrix(U, S, Vt, k)
    plt.subplot(1, 3, idx)
    plt.imshow(A_k, cmap='viridis', aspect='auto')
    plt.title(f'Approximation with k={k}')
    plt.colorbar()
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))

    plt.suptitle('SVD: Singular Values and Matrix Approximations')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Define the matrix A
A = np.array([[1, 0], 
              [0, 1], 
              [1, 1]])

# TODO: Call the visualization function to display results.
visualize_matrix_action(A)

