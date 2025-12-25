import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Sample Data
data = {'Square_Feet': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000],
    'Price': [400000, 450000, 500000, 600000, 650000, 700000, 750000, 800000]}
df = pd.DataFrame(data)
# Splitting the data into training and test sets
X = df[['Square_Feet']] # Independent variable (feature) 
y = df['Price'] 

#Dependent variable (target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Creating the model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)
# Plotting the results
plt.scatter(X_test, y_test, color='blue',label='Actual Prices') 
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.title('Linear Regression: Actual vs PredictedPrices') 
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.show()
# Print coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope: {model.coef_[0]}')