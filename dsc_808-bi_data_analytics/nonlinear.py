'''
Portfolio optimization An investor wants to allocate $100,000 between 
two stocks (X and Y) to maximize returns. The expected return for X is 8% and for Y is 10%. 
The risk constraint is that the variance of the portfolio must not exceed 0.02.
'''

from scipy.optimize import minimize

# Objective function (negative for maximization)
def objective(vars):
    x, y = vars
    return -(0.08 * x + 0.10 * y)  # Maximize return

# Constraints
def budget_constraint(vars):
    x, y = vars
    return x + y - 100000  # Total investment should equal $100,000

def risk_constraint(vars):
    x, y = vars
    return 0.0004 * x**2 + 0.0005 * y**2 - 0.02  # Risk constraint

# Bounds and initial guess
bounds = [(0, 100000), (0, 100000)]  # Bounds for x and y
initial_guess = [50000, 50000]  # Initial allocation
constraints = [
    {"type": "eq", "fun": budget_constraint},  # Budget constraint
    {"type": "ineq", "fun": risk_constraint}   # Risk constraint
]

# Solve the optimization problem
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

# Extract results
x, y = result.x
print(f"Optimal Allocation: Stock X: ${x:.2f}, Stock Y: ${y:.2f}")
print(f"Maximum Return: ${-result.fun:.2f}")