import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X = pd.DataFrame(X)
y = pd.Series(y)

for criteria in ['information_gain', 'gini_index']:
    tree = RandomForestClassifier
    Classifier_RF = RandomForestClassifier(10, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot()
    print('Criteria - Classifier:', criteria)
    print('Accuracy - Classifier: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision- Classifier: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))
