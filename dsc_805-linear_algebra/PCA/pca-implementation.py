from sklearn.decomposition import PCA
import numpy as np
# Example dataset
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
    [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])
# Standardizing the data (optional step)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("PCA Transformed Data: \n", X_pca)