# Importing necessary libraries
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Linear Regression model
linear_reg = LinearRegression()

# Training the model
linear_reg.fit(X_train, y_train)

# Making predictions on the test set
predictions = linear_reg.predict(X_test)

# Calculating metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
