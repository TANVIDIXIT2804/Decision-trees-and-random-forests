import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostedRegressor:
    def __init__(self, n_estimators=3, learning_rate=0.1, max_depth=2, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize regression trees
        self.trees = [DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                      for _ in range(n_estimators)]

    def fit(self, X, y):
        # Initialize the residual to the target variable
        self.residuals = y.copy()

        # Fit trees
        for i in range(self.n_estimators):
            tree = self.trees[i]

            # Update the residuals
            y_pred = self.predict(X, n_estimators=i)
            residual = y - y_pred
            self.residuals = residual

            # Fit the tree to the residuals
            tree.fit(X, residual)

    def predict(self, X, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators

        # Initialize predictions to 0
        y_pred = np.zeros(X.shape[0])

        # Make predictions with each tree
        for i in range(n_estimators):
            tree = self.trees[i]
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
