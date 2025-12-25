'''
Case study: Optimizing marketing budget allocation 
A company wants to allocate its marketing budget across three channels: 
social media, email marketing, and search engine advertising. 
The goal is to maximize the number of new customer acquisitions within the constraints of the available budget and channel-specific limitations.

1. Scenario The marketing team has a budget of $50,000 to spend across the three channels. The cost per acquisition (CPA) and the maximum spend limit for each channel are as follows:
Social media: $25 per customer, maximum spend $20,000
Email marketing: $20 per customer, maximum spend $15,000
Search engine advertising: $30 per customer, maximum spend $25,000
The objective is to determine the optimal allocation of the budget to maximize the total number of new customers.
'''

from pulp import LpMaximize, LpProblem, LpVariable

# Define the problem
model = LpProblem("Marketing_Budget_Optimization", LpMaximize)

# Decision variables
x1 = LpVariable("Social_Media", lowBound=0, upBound=20000)  # Budget for Social Media
x2 = LpVariable("Email_Marketing", lowBound=0, upBound=15000)  # Budget for Email Marketing
x3 = LpVariable("Search_Engine_Advertising", lowBound=0, upBound=25000)  # Budget for Search Engine Advertising

# Objective function: Maximize total customers acquired
model += (x1 / 25) + (x2 / 20) + (x3 / 30), "Total_Customers"

# Constraints
model += x1 + x2 + x3 <= 50000, "Total_Budget"  # Total budget constraint

# Solve the problem
model.solve()

# Output results
print("Optimal Allocation:")
print(f"  Social Media: ${x1.varValue:.2f}")
print(f"  Email Marketing: ${x2.varValue:.2f}")
print(f"  Search Engine Advertising: ${x3.varValue:.2f}")
print(f"Total Customers Acquired: {model.objective.value():.2f}")