import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Create a synthetic dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 8, 5, 9],
    'Attendance': [70,80,90,60,85,75,65,90,100,98, 70,50,99,100],
    'AssignmentCompletion': [0.5,0.8,1.0,0.7,1.0,0.9,0.2,1.0,0.95,1.0,0.7,0.4,1.0,1.0],
    'PassFail': ['Fail','Fail','Pass','Fail','Pass','Pass','Fail','Pass','Pass','Pass','Pass','Fail','Pass','Pass']
}
df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")
# Separate features and labels
X = df[['StudyHours', 'Attendance', 'AssignmentCompletion']]
y = df['PassFail']
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = dt_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Predicted labels: ", y_pred)
print("Actual labels: ", y_test.values)
print("Accuracy: {:.2f}%".format(accuracy * 100))