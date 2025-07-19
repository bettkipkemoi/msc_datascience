"""
A bookstore sells an average of 4 rare books per day. What is the probability that no rare books will be sold in a day?
Here, 𝜆=4 and 𝑘=2.

"""
from scipy.stats import poisson
lam, k = 4, 2
prob = poisson.pmf(k, lam)
print(f"{prob:.4f}")