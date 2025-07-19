import pandas as pd

# Assuming df is your Pandas DataFrame
# For demonstration purposes, create a sample DataFrame
data = {'col1': [1, 2, 3], 'col2': ['A', 'B', 'C'], 'col3': [True, False, True]}
df = pd.DataFrame(data)

# Get the number of rows and columns
num_rows, num_columns = df.shape

# Print the results
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")