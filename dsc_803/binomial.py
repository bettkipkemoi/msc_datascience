"""
In a class, 80.0% of students pass the exam. If 24 students are randomly selected, what is the probability that exactly 11 students pass?
Here, 𝑛=24, 𝑘=11, and 𝑝=0.8.

"""

from scipy.stats import binom
n, k, p = 24, 11, 0.8
prob = binom.pmf(k, n, p)
print(f"{prob:.4f}")