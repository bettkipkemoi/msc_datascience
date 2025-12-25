# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# Helper function to plot decision boundaries
def plot_decision_boundary(model, X, y, ax, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Define consistent color map
    # Colors picked to match screenshot style
    colors = ('#b3cde3', '#fbb4ae', '#ccebc5', '#decbe4', '#fed9a6')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot contour background
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=cmap)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=35, edgecolor='k', cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(*scatter.legend_elements(), title="Class")

# ---- First: Simple Binary Classification ----
X_binary, y_binary = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, n_classes=2, random_state=2
)

model_binary = LogisticRegression()
model_binary.fit(X_binary, y_binary)

# ---- Second: Multi-Class Classification ----
X_multi, y_multi = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, n_classes=4, random_state=42
)

model_multi = LogisticRegression(multi_class='ovr', max_iter=200)
model_multi.fit(X_multi, y_multi)

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_decision_boundary(model_binary, X_binary, y_binary, axes[0],
                       "Simple Classification Example (2 classes)")
plot_decision_boundary(model_multi, X_multi, y_multi, axes[1],
                       "Multi-Class Classification (4 Classes)")

plt.tight_layout()
plt.show()