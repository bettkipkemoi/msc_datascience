import pandas as pd

data = {
    'age': [25, 30, 28, 40, 35, 38, 27, 45, 50, 29],
    'income': ['high', 'high', 'medium', 'low', 'low', 'medium', 'low', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'excellent', 'fair', 'fair', 'excellent', 'fair', 'excellent'],
    'Purchased_product': ['no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes']
}

df_sales = pd.DataFrame(data)

# X: all columns except "Purchased_product"
X = df_sales.drop(columns=['Purchased_product'])

# y: only the "Purchased_product" column
y = df_sales['Purchased_product']

# For quick verification when running this file directly
if __name__ == "__main__":
    print("X columns:", list(X.columns))
    print("y values:", y.tolist())