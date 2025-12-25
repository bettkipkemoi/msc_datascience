import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Step 1: Generate Synthetic Data
np.random.seed(42)
education = np.random.normal(16, 2, 1000)  # Education levels, mean=16 years
income = 5 * education + np.random.normal(0, 10, 1000)  # Income correlated with education

# Create a DataFrame
data = pd.DataFrame({'Education': education, 'Income': income})

# Step 2: Plot the Joint Distribution
sns.jointplot(data=data, x='Education', y='Income', kind='scatter', color='blue', alpha=0.6)
plt.suptitle("Joint Distribution of Income vs. Education", y=1.02)
plt.show()

# Step 3: 2D KDE (Kernel Density Estimation) for Joint Distribution
x, y = data['Education'], data['Income']
kde = gaussian_kde([x, y])
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                             np.linspace(y.min(), y.max(), 100))
z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, z, levels=20, cmap='viridis')
plt.colorbar(label='Density')
plt.title("2D Kernel Density Estimation")
plt.xlabel("Education (Years)")
plt.ylabel("Income ($1000s)")
plt.show()

# Step 4: Correlation Analysis
correlation = data.corr()
print("Correlation Matrix:")
print(correlation)