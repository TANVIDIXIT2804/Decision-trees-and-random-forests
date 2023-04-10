from typing import List
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''

        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1, criterion = 'entropy')

        self.baseEstimator: DecisionTreeClassifier = base_estimator
        self.estimators_list = [self.baseEstimator(max_depth = 1, criterion = 'entropy') for _ in range(n_estimators)]
        self.nEstimators = n_estimators

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.trainX = X
        self.trainY = y
        weights = np.ones(len(X)) / len(X)
        alpha_list = []
        self.weights = []
        for i, estimator in enumerate(self.estimators_list):
            self.weights.append(weights.copy())
            estimator.fit(X, y, sample_weight=weights.copy())

            predictions = estimator.predict(X)
            error = np.sum(weights[predictions != y]) / np.sum(weights) # no need to normalize since its already summing to 1
            alpha = 0.5 * np.log((1 - error)/error)

            alpha_list.append(alpha)
            wrong = predictions != y
            correct = predictions == y
            weights[wrong] = weights[wrong] * np.exp(alpha)
            weights[correct] = weights[correct] * np.exp(-alpha)

            weights /= np.sum(weights)
            
            

        self.alpha_list = np.array(alpha_list)
        self.weights = np.array(self.weights)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        self.testX = X
        preds = np.array([estimator.predict(X) for estimator in self.estimators_list])*2 - 1
        y = pd.Series(((np.sign(np.dot(preds.T, self.alpha_list)) + 1)/2).astype(int))


        return y
    
    def plotOneEstimator(self, ax, predict, xx, yy, X, y, weights):

        # calling the predict method of the given estimator object on the flattened mesh grid
        Z = predict(np.c_[xx.ravel(), yy.ravel()])
        if type(Z) != np.ndarray:
            Z = Z.to_numpy()
        Z = Z.reshape(xx.shape) #reshaped to shape of mesh grid
        ax.contourf(xx, yy, Z, cmap= plt.cm.RdBu, alpha=0.8)

        # Plot the training points
        ax.scatter(
            X[:, 0], X[:, 1], c=y, s = weights, cmap=ListedColormap(["#FF0000", "#0000FF"]), edgecolors="k"
        )
      
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # ax.set_xticks(())
        # ax.set_yticks(())
        
    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        h = 0.02 #resolution of decision boundary plot 
        X_train = self.trainX.to_numpy() #convert to numpy array for plotting
        y_train = self.trainY.to_numpy()
        
        #setting the minimum and maximum values of the x-axis and y-axis
        #for the plot based on the range of the first feature in the training data X
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

        # creates a grid of coordinates (xx and yy) that covers the entire x-y range 
        # of the training data, with a step size of h.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        fig1, ax = plt.subplots(figsize=(20, 5))
        
        for i in range(self.nEstimators):
            ax = plt.subplot(1, self.nEstimators, i+1)
            plt.title(f'Estimator {i}, alpha = {self.alpha_list[i]}')
            weights = 20 * 2**( 2*(self.weights[i])/max(self.weights[i]))
            #plotting the decision boundary of the current base model on the current subplot
            self.plotOneEstimator(ax, self.estimators_list[i].predict, xx, yy, X_train, y_train, weights)

        fig2, ax = plt.subplots()
        plt.title(f'Final Estimator')
        self.plotOneEstimator(ax, self.predict, xx, yy, X_train, y_train, np.ones(len(X_train)) * 80 )
            

    
        return fig1, fig2
