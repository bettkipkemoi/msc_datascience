import numpy as np
import matplotlib.pyplot as plt

# Function to compute the pseudoinverse and solve the least squares problem
def least_squares_pseudoinverse(A, b):
    A_pseudo = np.linalg.pinv(A)
    x = A_pseudo @ b
    return x

# Example system
A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2])

# Solve the least squares problem
x_least_squares = least_squares_pseudoinverse(A, b)
print("Least Squares Solution:", x_least_squares)

# Visualization
x_values = np.linspace(0, 4, 100)
y_fitted = x_least_squares[0] + x_least_squares[1] * x_values
y_pred = A @ x_least_squares

# Plot data points and fitted line
plt.scatter(A[:, 1], b, color='red', label="Data points")
plt.plot(x_values, y_fitted, label="Fitted Line", color='blue')

# Plot residuals
for i in range(len(b)):
    plt.vlines(A[i, 1], b[i], y_pred[i], color='green', linestyle='--')

plt.title("Least Squares Fit with Residuals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()