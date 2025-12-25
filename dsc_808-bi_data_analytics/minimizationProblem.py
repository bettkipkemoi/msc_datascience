'''
Facility location A company wants to open two warehouses in three cities 
(A, B, C) to minimize costs. The setup costs are $20,000 for City A, 
$25,000 for City B, and $30,000 for City C. The company must open exactly two warehouses.
'''

from pulp import LpMinimize, LpProblem, LpVariable

# Define the problem
problem = LpProblem("Facility_Location", LpMinimize)

# Decision variables
y1 = LpVariable("City_A", cat="Binary")  # Binary variable for City A
y2 = LpVariable("City_B", cat="Binary")  # Binary variable for City B
y3 = LpVariable("City_C", cat="Binary")  # Binary variable for City C

# Objective function: Minimize total cost
problem += 20000 * y1 + 25000 * y2 + 30000 * y3, "Total Cost"

# Constraints
problem += y1 + y2 + y3 == 2, "Open two warehouses"

# Solve the problem
problem.solve()

# Output results
print("Optimal Locations:")
print(f"  City A: {'Open' if y1.varValue == 1 else 'Closed'}")
print(f"  City B: {'Open' if y2.varValue == 1 else 'Closed'}")
print(f"  City C: {'Open' if y3.varValue == 1 else 'Closed'}")
print(f"Minimum Cost: ${problem.objective.value():.2f}")