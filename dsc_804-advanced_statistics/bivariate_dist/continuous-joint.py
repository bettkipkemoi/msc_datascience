import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Mean and covariance matrix
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Create a grid of (x, y) values
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Evaluate the PDF
rv = multivariate_normal(mean, cov)
Z = rv.pdf(pos)

# Contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, cmap='viridis', levels=30)
plt.colorbar(contour)
plt.title("Contour Plot of Bivariate Normal Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
ax.set_title("3D Surface Plot of Bivariate Normal Distribution")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Probability Density")
plt.show()
