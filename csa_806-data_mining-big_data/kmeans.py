import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq

# Step 1: Generate synthetic sample dataset (200 customers)
np.random.seed(42)  # For reproducibility
num_customers = 200
categories = ['Groceries', 'Clothing', 'Electronics', 'Entertainment', 'Health']

data = {cat: np.random.randint(1000, 50000, num_customers) for cat in categories}
df = pd.DataFrame(data)

print("Sample dataset head:")
print(df.head())

# Step 2: Normalize the data (standardization)
data_normalized = (df[categories] - df[categories].mean()) / df[categories].std()

# Step 3: Elbow method to find optimal k
distortions = []
K = range(1, 11)
for k in K:
    centroids, distortion = kmeans(data_normalized.values, k)
    distortions.append(distortion)

print("\nDistortions for k=1 to 10:", distortions)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Step 4: Apply k-Means with k=4 (based on elbow)
k = 4
centroids, _ = kmeans(data_normalized.values, k)
cluster_labels, _ = vq(data_normalized.values, centroids)
df['Cluster'] = cluster_labels

# Step 5: Analyze results
print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())

cluster_means = df.groupby('Cluster')[categories].mean()
print("\nAverage spending per cluster (in KES):")
print(cluster_means)

# Visualize average spending by cluster
cluster_means.plot(kind='bar', figsize=(10, 6))
plt.title('Average Annual Spending by Customer Segment')
plt.ylabel('Average Spending (KES)')
plt.xlabel('Cluster')
plt.legend(title='Categories')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()