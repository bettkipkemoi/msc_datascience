import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# Step 1: Scatter Plot
sns.jointplot(data=data, x='sepal_length', y='petal_length', kind='scatter', color='green', alpha=0.6)
plt.suptitle("Joint Distribution of Sepal Length vs. Petal Length", y=1.02)
plt.show()

# Step 2: 2D KDE Plot
x, y = data['sepal_length'], data['petal_length']
kde = gaussian_kde([x, y])
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                             np.linspace(y.min(), y.max(), 100))
z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, z, levels=20, cmap='viridis')
plt.colorbar(label='Density')
plt.title("2D Kernel Density Estimation")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# Step 3: Correlation Analysis
correlation = data[['sepal_length', 'petal_length']].corr()
print("Correlation Matrix:")
print(correlation)