'''
An investor wants to allocate $1,000,000 between two assets, 
Asset A and Asset B, to maximize expected returns. 
The data for the two assets is as follows:

Asset A: Expected return = 8%, Variance = 0.02.
Asset B: Expected return = 12%, Variance = 0.04.
Covariance between Asset A and Asset B = 0.01.

The investor wants to ensure that the portfolio risk (variance) does not exceed 0.03. 
The objective is to find the optimal allocation between Asset A and Asset B to maximize the portfolio’s expected return.
'''

from scipy.optimize import minimize

# Objective function (negative because we maximize)
def objective(vars):
    x1, x2 = vars
    return -(0.08 * x1 + 0.12 * x2)

# Constraints
def total_allocation(vars):
    x1, x2 = vars
    return x1 + x2 - 1

def risk_constraint(vars):
    x1, x2 = vars
    return 0.03 - (0.02 * x1**2 + 0.04 * x2**2 + 0.01 * x1 * x2)

# Bounds and initial guess
bounds = [(0, 1), (0, 1)]
initial_guess = [0.5, 0.5]
constraints = [
    {"type": "eq", "fun": total_allocation},
    {"type": "ineq", "fun": risk_constraint}
]

# Solve the optimization problem
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
x1, x2 = result.x

# Output results
print(f"Optimal Allocation: Asset A = {x1:.2%}, Asset B = {x2:.2%}")
print(f"Maximum Expected Return: {-result.fun:.2%}")