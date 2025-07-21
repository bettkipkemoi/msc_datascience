"""
Perform Principal Component Analysis (PCA) the generated dataset to reduce its dimensionality and visualize the result.
"""
import numpy as np
from sklearn . decomposition import PCA
import matplotlib.pyplot as plt
# Generate a dataset
np . random . seed (42)
data = np . random . rand (100 , 5)
#TODO:  Apply PCA to reduce to 2 dimensions
principal_components = PCA(n_components=2)
X_pca = principal_components.fit_transform(data)
# TODO:Plotting PCA results
plt.scatter( X_pca [ : , 0 ] , X_pca [ : , 1 ] )
plt.title( "PCA with random data")
plt.xlabel( "Principal Components 1")
plt.ylabel( "Principal Components 1" )
plt.show()