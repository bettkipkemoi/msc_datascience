'''
Production planning A company produces two products, A and B. 
Each unit of A requires 2 hours of labor and 3 units of material. 
Each unit of B requires 1 hour of labor and 2 units of material. 
The company has 100 labor hours and 120 units of material available. 
Each unit of A generates a profit of $40, and each unit of B generates a profit of $30. 
The objective is to maximize profit.
'''

# Import necessary modules
from pulp import LpMaximize, LpProblem, LpVariable

# Define the problem
problem = LpProblem("Production_Planning", LpMaximize)

# Decision variables
x1 = LpVariable("Product_A", lowBound=0, cat="Continuous")  # Units of Product A
x2 = LpVariable("Product_B", lowBound=0, cat="Continuous")  # Units of Product B

# Objective function: Maximize profit
problem += 40 * x1 + 30 * x2, "Profit"

# Constraints
problem += 2 * x1 + x2 <= 100, "Labor"        # Labor constraint
problem += 3 * x1 + 2 * x2 <= 120, "Material"  # Material constraint

# Solve the problem
problem.solve()

# Output results
print("Optimal Production Plan:")
print(f"  Product A: {x1.varValue:.2f} units")
print(f"  Product B: {x2.varValue:.2f} units")
print(f"Maximum Profit: ${problem.objective.value():.2f}")