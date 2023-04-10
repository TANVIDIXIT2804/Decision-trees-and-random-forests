import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
# from tree.base import DecisionTree

# Or use sklearn decision tree
from sklearn.tree import DecisionTreeClassifier
########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

Classifier_B = BaggingClassifier(base_estimator=DecisionTreeClassifier, n_estimators=n_estimators )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot()
print('Criteria :', 'entropy')
print('Accuracy: ', accuracy(y_hat, y))
print(y_hat.to_numpy(), )
print(y.to_numpy())
for cls in y.unique():
    print()
    print('Class', cls)
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

# import multiprocessing
# import time

# def task():
#     # print('Sleeping')
#     BaggingClassifier(base_estimator=DecisionTreeClassifier, n_estimators=n_estimators)
#     # print('Finished sleeping')

# if __name__ == "__main__":
#     n_jobs=10
#     start_time = time.perf_counter()
#     processes=[]
#     # Creates two processes
#     for i in range(n_jobs):
#       p=multiprocessing.Process(target=task)
#       p.start()
#       processes.append(p)
#     # Starts both processes
#     for p in processes:
#       p.join()

#     finish_time = time.perf_counter()

#     print(f"Program finished in {finish_time-start_time} seconds")
