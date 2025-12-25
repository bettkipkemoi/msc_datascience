# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create the MLPClassifier model and initialize with max_iter=1000 and random_state=0
model = MLPClassifier(max_iter=1000, random_state=0)

# Fit the model on training data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate and print accuracy score 
print(accuracy_score(y_test, predictions))