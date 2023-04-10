from tree.base import DecisionTree

from sklearn.tree import DecisionTreeClassifier

## compare both the trees
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap

# Generate the classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
X = pd.DataFrame(X)
# print(type(y))

y = pd.Series(y)
y = y.astype('category')
# Shuffle and split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#print(X_test.shape)

# Assign random weights to the training samples
sample_weights = np.random.uniform(0, 1, size=len(X_train))

# Create a weighted decision tree model
model = DecisionTree(max_depth=4, sample_weights = sample_weights)
model.fit(X_train, y_train, sample_weights)

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = y_pred.reset_index(drop = True)

# Check accuracy
accuracy = accuracy_score(y_test.reset_index(drop = True), y_pred.reset_index(drop = True))
print('My Decision Tree Accuracy:', accuracy)

# Compare with scikit-learn's decision tree
sk_model = DecisionTreeClassifier(max_depth=4, class_weight='balanced')
sk_model.fit(X_train, y_train, sample_weights)
sk_y_pred = sk_model.predict(X_test)
sk_accuracy = accuracy_score(y_test, sk_y_pred)
print('Scikit-learn Decision Tree Accuracy:', sk_accuracy)

# # Plot the decision boundary
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(X_test, y_pred, clf=model, legend=2)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Weighted Decision Tree Boundary')
# plt.show()




