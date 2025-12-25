# Logistic Regression with Visuals on the Iris Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 1️⃣ Scatter Plot of Predictions (PCA projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_pred, cmap='viridis', s=60, edgecolor='k')
plt.title("Predicted Iris Classes (Logistic Regression)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# 2️⃣ Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=load_iris().target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Logistic Regression on Iris")
plt.show()

# 3️⃣ Decision Boundary Visualization
X_reduced = pca.fit_transform(X)
model.fit(X_reduced, y)

x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=40, edgecolor='k')
plt.title("Decision Boundaries for Iris (Logistic Regression)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()