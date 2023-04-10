from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
import matplotlib.pyplot as plt
from statistics import mean

import numpy as np
import pandas as pd
import random

from operator import itemgetter

np.random.seed(1234)
xi = np.linspace(0, 10, 50)
x = (xi-0)/(9-0) # normalize
epsi = np.random.normal(0, 5, 50)
eps = (epsi - 0)/(4-0) # normalize
y = x**2 +1 + eps

length_x = len(x)
models = 5
sample_size = 30

max_depth =[i for i in range(1, 10)]

Range = [i for i in range(0, 50)]
indices ={}
#y_pred ={}
variance =[]
bias =[]
for i in range(models):
    np.random.seed(i)
    indices[i] = np.random.choice(Range, sample_size)

for depth in max_depth:
    y_pred ={}
    
    for i in range(models):
        X = np.array(x)
        X_train = np.array(itemgetter(*indices[i])(x))
        Y_train = np.array(itemgetter(*indices[i])(y))
        
        
        #X_train, X_test, y_train, y_test = train_test_split(X.reshape(sample_size,-1).tolist(), Y, test_size=0.2, random_state=77)
        model = DecisionTreeRegressor(max_features=1, max_depth=depth).fit(X_train.reshape(sample_size,-1).tolist(), Y_train)
        
        
        Y_pred = model.predict(X.reshape(50,-1).tolist())
        y_pred[i] = Y_pred
        
    y_pred = pd.DataFrame(y_pred)
   
    y_pred["variance"] = y_pred.var(axis=1)
    y_pred["mean"]= y_pred.mean(axis=1)
    y_pred["y true"] = y
    bias.append(mean_absolute_error(y_pred["mean"], y_pred["y true"])**2)
    variance.append(np.mean(y_pred["variance"]))
    
        
plt.plot(max_depth, bias, label='Bias')
plt.plot(max_depth, variance, label='Variance')
plt.legend()
plt.xlabel('Maximum Depth')
plt.ylabel('Bias/Variance')
plt.title('Bias-Variance Tradeoff for decision trees')
plt.show()
