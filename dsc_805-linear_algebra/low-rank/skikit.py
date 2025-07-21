from sklearn.decomposition import TruncatedSVD
import numpy as np
# Example dataset (with higher dimensions)
X_s = np.random.rand(10, 10)
# Apply TruncatedSVD
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X_s)
print("Original dataset shape:", X_s.shape)
print("Reduced dataset shape:", X_reduced.shape)