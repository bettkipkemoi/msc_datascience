# Set seed for reproducibility
set.seed(123)

# Number of samples
n <- 1000

# Degrees of freedom
df <- 10

# Simulate data from t-distribution
t_data <- rt(n, df)

# Display first few simulated values
head(t_data)
