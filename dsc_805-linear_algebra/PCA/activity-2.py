"""
In this activity, you will be required to:

Compute the covariance matrix of a dataset.
Understand how variance is represented.
Explore the relationship between covariance and PCA.
Complete the code template provided and submit your solution.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA

# Step 1: Generate the Dataset
np.random.seed(42)
cov_matrix = [[1.0, 0.8, 0.6],
              [0.8, 1.0, 0.7],
              [0.6, 0.7, 1.0]]
mean = [0, 0, 0]
size = 100
data = np.random.multivariate_normal(mean=mean, cov=cov_matrix, size=size)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])

# Step 2: Compute the Covariance Matrix of the Dataset
computed_cov_matrix = df.cov().values


# Step 3: Perform PCA to Compute Eigenvalues and Eigenvectors
# Perform PCA using sklearn
pca = PCA(n_components=2)
pca.fit(df)

# Eigenvalues and eigenvectors from the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(computed_cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]


# Step 4: Visualize Original Data and PCA Transformed Data
plt.figure(figsize=(10, 5))

# Original data visualization
plt.subplot(1, 2, 1)
plt.scatter(df['Feature1'], df['Feature2'], alpha=0.7, label='Original Data')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Original Data (Feature1 vs Feature2)')
plt.legend()

# Transformed data visualization
plt.subplot(1, 2, 2)
transformed = pca.transform(df)
plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.7, color='orange', label='PCA Transformed')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformed Data')
plt.legend()

plt.tight_layout()
plt.show()


# Step 5: Save Results to JSON File
output = {
    "cov_matrix": computed_cov_matrix.tolist(),
    "eigenvalues": eigenvalues.tolist(),
    "visualization": "done"
}

# Write output to student_output.json
with open("student_output.json", "w") as f:
    json.dump(output, f, indent=4)

print("Results saved to student_output.json.")
