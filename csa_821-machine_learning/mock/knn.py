# Step 1: Import required modules
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Step 3: Split the dataset (80% train, 20% test) with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Step 4: Create and train the KNN model with 3 neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Print accuracy to verify
print("KNN accuracy:", accuracy_score(y_test, y_pred))