from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data
y = iris.target



#Split the data to train and test data using an 80/20 
#training/testing split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


###Create the logistic regressor and call it logistic_model

logistic_model = LogisticRegression(max_iter=1000, random_state=0)



### Fit the  Model
logistic_model.fit(X_train, y_train)

# Make predictions and print accuracy
preds = logistic_model.predict(X_test)
print("Logistic Regression accuracy:", logistic_model.score(X_test, y_test))

###Create the logistic regressor and call it logistic_model






### Fit the  Model