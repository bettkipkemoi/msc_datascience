import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3, 2], [2, 3], [1, 1]])
U, S, Vt = np.linalg.svd(A)

# Visualizing transformation
v = np.array([1, 0])
u = A @ v

plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='b', label='Av')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid()
plt.legend()
plt.show()