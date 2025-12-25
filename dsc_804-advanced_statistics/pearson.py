'''
Given a Pandas DataFrame df with columns 'Height' and 'Weight', write a code snippet to calculate the Pearson correlation coefficient between them and the Covariance matrix.
Your code should print:
"Correlation: {correlation_value}"
"Covariance Matrix:\n{covariance_matrix}"
'''

import pandas as pd
df = pd.DataFrame({'Height': [160, 170, 180], 'Weight': [60, 70, 80]})

correlation_value = df['Height'].corr(df['Weight'])
covariance_matrix = df.cov()
print(f"Correlation: {correlation_value}")
print(f"Covariance Matrix:\n{covariance_matrix}")