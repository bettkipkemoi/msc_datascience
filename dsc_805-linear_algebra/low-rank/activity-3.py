import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon

# TODO: Plot unit balls in different norms
fig, ax = plt.subplots(figsize=(6, 6))

# TODO: Define the $\ell_2$ norm (circle)
circle_1 = Ellipse((0, 0), width=2, height=2, edgecolor='blue', facecolor='none', lw=2, label='$\\ell_2$ (Euclidean)')

# TODO:Define the $\ell_\infty$ norm (square)
square_1 = Rectangle((-1, -1), 2, 2, edgecolor='green', facecolor='none', lw=2, label='$\\ell_\\infty$ (Max)')

# TODO:Define the $\ell_1$ norm (diamond)
diamond_1 = Polygon([[0, 1], [1, 0], [0, -1], [-1, 0]], closed=True, edgecolor='red', facecolor='none', lw=2, label='$\\ell_1$ (Manhattan)')

# Add shapes to the plot
ax.add_patch(circle_1)
ax.add_patch(square_1)
ax.add_patch(diamond_1)

# Set limits and aspect ratio
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal', adjustable='box')

# Add legend
ax.legend()

# Display the plot
plt.show()