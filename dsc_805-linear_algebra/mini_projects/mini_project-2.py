import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define linear transformations
def scale_matrix(sx, sy):
    """Scale matrix for 2D transformations."""
    #TODO
    return np.array([
        [sx, 0],
        [0, sy]
    ])

def rotation_matrix(theta):
    """Rotation matrix for 2D transformations."""
    #TODO
    theta_rad = np.deg2rad(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])

def shear_matrix(shx, shy):
   #TODO
    return np.array([
        [1, shx],
        [shy, 1]
    ])

def apply_transformation(matrix, vectors):
    """Apply matrix transformation to a set of vectors."""
   #TODO
    # The matrix should be 2x2, vectors should be Nx2 (each row is a vector)
    # Return the transformed vectors (same shape as input)
    return vectors @ matrix.T

# Create a grid of points (for visualization)
def create_grid(x_range, y_range, step=0.5):
    """Generate a grid of points."""
    #TODO
    # Generate a grid of points in 2D space within the given x and y ranges
    x_vals = np.arange(x_range[0], x_range[1] + step, step)
    y_vals = np.arange(y_range[0], y_range[1] + step, step)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return grid

# Visualization function
def plot_transformation(vectors, transformed_vectors, ax, title="Linear Transformation"):
    ax.clear()
    ax.quiver(vectors[:, 0], vectors[:, 1], angles='xy', scale_units='xy', scale=1, color='b', label="Original Vectors")
    ax.quiver(transformed_vectors[:, 0], transformed_vectors[:, 1], angles='xy', scale_units='xy', scale=1, color='r', label="Transformed Vectors")
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

def update(val):
    """Update the transformation based on slider values."""
    try:
        sx = slider_sx.val
        sy = slider_sy.val
        theta = slider_theta.val
        shx = slider_shx.val
        shy = slider_shy.val
        
        # Create transformation matrix
        matrix = scale_matrix(sx, sy) @ rotation_matrix(theta) @ shear_matrix(shx, shy)
        
        # Apply transformation to the grid
        transformed_grid = apply_transformation(matrix, grid_points)
        
        # Update plot
        plot_transformation(grid_points, transformed_grid, ax)
        plt.draw()
    except Exception as e:
        print(f"Error in update function: {e}")

# Create a set of points (e.g., vectors) and a grid
grid_points = create_grid((-5, 5), (-5, 5))

# Create the figure and axis for the plot
fig, ax = plt.subplots(figsize=(8, 8))
plot_transformation(grid_points, grid_points, ax)

# Add sliders for parameters
ax_slider_sx = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_sy = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_theta = plt.axes([0.15, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_shx = plt.axes([0.15, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_shy = plt.axes([0.15, 0.21, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_sx = Slider(ax_slider_sx, 'Scale X', 0.1, 3.0, valinit=1)
slider_sy = Slider(ax_slider_sy, 'Scale Y', 0.1, 3.0, valinit=1)
slider_theta = Slider(ax_slider_theta, 'Rotation (deg)', -180, 180, valinit=0)
slider_shx = Slider(ax_slider_shx, 'Shear X', -2.0, 2.0, valinit=0)
slider_shy = Slider(ax_slider_shy, 'Shear Y', -2.0, 2.0, valinit=0)

# Attach update function to sliders
slider_sx.on_changed(update)
slider_sy.on_changed(update)
slider_theta.on_changed(update)
slider_shx.on_changed(update)
slider_shy.on_changed(update)

# Display the interactive plot
plt.show()