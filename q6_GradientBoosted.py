import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from metrics import *

from ensemble.gradientBoosted import GradientBoostedClassifier
from tree.base import DecisionTree

# Or use sklearn decision tree

########### GradientBoostedClassifier ###################
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)

np.savetxt("regression_dataset.csv", np.concatenate([X, y.reshape(-1, 1)], axis=1), delimiter=",")

# Load dataset
data = pd.read_csv("regression_dataset.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = GradientBoostedRegressor(n_estimators=3, learning_rate=0.1, max_depth=2, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate mean squared error
Rmse = rmse(y_test, y_pred)
print("Root Mean Squared Error:", Rmse)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Gradient Boosted Decision Trees")
plt.show()
