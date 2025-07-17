import numpy as np
import pandas as pd

# Your code here
# Set random seed for reproducibility
np.random.seed(123)

# Generate 1000 normally distributed scores (mean=70, std=10)
scores = np.random.normal(loc=70, scale=10, size=1000)

# Print statistics
print(f"Mean: {np.mean(scores):.2f}")
print(f"Standard Deviation: {np.std(scores):.2f}")
print(f"Min: {np.min(scores):.2f}")
print(f"Max: {np.max(scores):.2f}")

# Adjust mean to 65 and regenerate scores
adjusted_scores = np.random.normal(loc=65, scale=10, size=1000)

# Print new mean
print(f"Adjusted Mean: {np.mean(adjusted_scores):.2f}")

# Save adjusted scores to CSV
df = pd.DataFrame(adjusted_scores, columns=['Score'])
df.to_csv('simulated_scores.csv', index=False)
print("Saved to simulated_scores.csv")