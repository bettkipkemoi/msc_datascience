# question
'''
A machine is claimed to fill bottles with 500 ml of liquid. 
A random sample of 40 bottles has a mean volume of 495 ml 
and a known population standard deviation of 10 ml. 
Test whether the machine is under filling bottles at \alpha = 0.01 using a p-value approach.
'''
# Import required libraries
from math import sqrt
from scipy.stats import norm

# Given values
n = 40               # Sample size
x_bar = 495          # Sample mean
mu = 500             # Population mean
sigma = 10           # Population standard deviation
alpha = 0.01         # Significance level

# Compute the Z-value
z_value = (x_bar - mu) / (sigma / sqrt(n))

# Compute the p-value for a one-tailed test
p_value = norm.cdf(z_value)  # CDF gives the area to the left of Z

# Decision rule using p-value
if p_value < alpha:
    print("Reject the null hypothesis: The machine is underfilling bottles.")
else:
    print("Fail to reject the null hypothesis: No evidence of underfilling.")

# Output the Z-value and p-value
print("Z-value:", z_value)
print("P-value:", p_value)


# explanation
'''
Explanation:
The null hypothesis H_0 is that the average volume is 500 ml.
The alternative hypothesis H_1 is that the machine is underfilling, i.e., 
the average volume is less than 500 ml.
The Z-value is calculated as:

z = (495 - 500) / (10 / √40) = -3.1623
 
The p-value is obtained using the cumulative distribution function pnorm(z_ value), 
which gives the probability of observing a Z-value as extreme or 
more extreme than the observed one under the null hypothesis.
If the p-value is less than the significance level (\alpha = 0.01), we reject the null hypothesis.

Interpretation:
- If the p-value= 0.0007827011 is small (e.g., less than \alpha = 0.01), 
we have strong evidence to reject the null hypothesis, indicating that the machine is underfilling.
'''