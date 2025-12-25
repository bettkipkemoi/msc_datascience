'''
A survey claims that 60% of customers are satisfied with a company’s service. 
A sample of 200 customers is taken, and 130 of them report being satisfied. 
Use a significance level of 𝛼=0.05to test if the satisfaction rate differs from 60%.
'''

import scipy.stats as stats
import math

# Given data
n = 200          # sample size
x = 130          # number satisfied
p_hat = x / n    # sample proportion
p0 = 0.60        # hypothesized proportion
alpha = 0.05     # significance level

# Step 1: Compute the standard error
se = math.sqrt(p0 * (1 - p0) / n)

# Step 2: Compute the z-statistic
z = (p_hat - p0) / se

# Step 3: Compute the two-tailed p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

# Step 4: Critical value for two-tailed test
z_critical = stats.norm.ppf(1 - alpha / 2)

# Results
print(f"Sample proportion (p̂): {p_hat:.4f}")
print(f"Test statistic (z): {z:.4f}")
print(f"Critical values: ±{z_critical:.4f}")
print(f"p-value: {p_value:.4f}")

if abs(z) > z_critical or p_value < alpha:
    print("Reject the null hypothesis: evidence that satisfaction rate differs from 60%.")
else:
    print("Fail to reject the null hypothesis: insufficient evidence that satisfaction rate differs from 60%.")