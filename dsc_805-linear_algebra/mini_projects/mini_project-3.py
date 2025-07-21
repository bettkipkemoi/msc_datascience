import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as la

# Function for LU decomposition
def lu_decomposition(A):
    #TODO
    P, L, U = la.lu(A)
    return P, L, U

# Function for QR decomposition
def qr_decomposition(A):
   #TODO
   Q, R = la.qr(A)
   return Q, R

# Function for Cholesky decomposition
def cholesky_decomposition(A):
    #TODO
    L = np.linalg.cholesky(A)
    return L

# Function to visualize matrices using heatmaps
def plot_matrix(matrix, title="Matrix", ax=None):
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    #TODO
    if ax is None:
        plt.figure(figsize=(4, 4))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(title)
        plt.show()
    else:
        ax.set_title(title)

# Function to compare decompositions
def visualize_decompositions(A):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # LU Decomposition
    P, L, U = lu_decomposition(A)
    #TODO
    plot_matrix(P, title="LU P Matrix", ax=axs[0, 0])
    plot_matrix(L, title="LU L Matrix", ax=axs[0, 1])
    plot_matrix(U, title="LU U Matrix", ax=axs[0, 2])
    
    # QR Decomposition
    Q, R = qr_decomposition(A)
   #TODO
    plot_matrix(Q, title="QR Q Matrix", ax=axs[1, 0])
    plot_matrix(R, title="QR R Matrix", ax=axs[1, 1])
    axs[1, 2].axis('off')  # Leave the last QR subplot empty (Cholesky will use it if possible)
    
    # Cholesky Decomposition (only works for positive definite matrices)
    if np.all(np.linalg.eigvals(A) > 0):  # Check if A is positive definite
        L = cholesky_decomposition(A)
        plot_matrix(L, title="Cholesky L Matrix", ax=axs[1, 2])
    else:
        axs[1, 2].text(0.5, 0.5, 'Not Positive Definite', ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Example matrix (positive definite for Cholesky)
A = np.array([[4, 2], [2, 3]])

# Visualize the decompositions
visualize_decompositions(A)
