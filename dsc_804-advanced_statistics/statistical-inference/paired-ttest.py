from scipy.stats import ttest_rel

# Data
before = [60, 62, 64, 58, 66, 64, 68, 70]
after = [65, 67, 69, 60, 71, 66, 72, 74]

# Perform paired t-test (one-tailed test for an increase)
t_stat, p_value = ttest_rel(after, before, alternative='greater')

# Output the results
print("Paired t-test")
print(f"t = {t_stat:.3f}")
print(f"p-value = {p_value:.5e}")

# Decision rule
alpha = 0.05  # Significance level
if p_value < alpha:
    print("Reject the null hypothesis: There is evidence of an increase.")
else:
    print("Fail to reject the null hypothesis: No evidence of an increase.")

# conclusion
'''
Since the p-value = 2.77723e-05, which is much smaller than 0.05, 
we reject the null hypothesis, concluding that the training program improves employee performance.
'''