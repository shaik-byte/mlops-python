# Importing necessary libraries
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Logistic Regression model
logistic_reg = LogisticRegression()

# Training the model
logistic_reg.fit(X_train, y_train)

# Making predictions on the test set
predictions = logistic_reg.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
# print("Accuracy:", accuracy)
