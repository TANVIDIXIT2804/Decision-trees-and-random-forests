"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index
from dataclasses import dataclass
from typing import Literal

np.random.seed(42)

class DecisionTree():
    def __init__(self, sample_weights, criterion = 'information_gain', max_depth=np.inf):
        """
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """

        # gini score ~~ info gain except that entropy replaced with gini index
        if criterion == "information_gain":
            self.criterion = entropy
        else:
            self.criterion = gini_index

        self.maxDepth = max_depth
        self.tree: dict = None

    # getting input for discrete input based on ig
    def computeRoot(self, X, y, next_attribute):
        gains = []

        for attr in next_attribute :
            g = information_gain(y, X[attr], sample_weights, self.criterion)
            gains.append(g)

        next_best = next_attribute[np.argmax(gains)]
        return next_best

    # for Discrete input Discrete output case;
    def DIDO(self, sample: pd.DataFrame, sample_value: pd.Series, next_attr, depth=0, base_attr=None):
      if len(sample_value.drop_duplicates()) == 1 or len(sample.drop_duplicates()) == 1 or depth >= self.maxDepth:
        #if there is a single value in dataframe or a single type of data outputs or the max depth has been reached 
        #then it will return the value that's the maximum times found in the output column
        return sample_value.value_counts().idxmax()

      # if we have reached the end of the tree i.e. incase of no more attributes to split on
      elif len(next_attr) == 0  or len(sample_value) == 0:  
        return base_attr

      base_attr = self.computeRoot(sample, sample_value, next_attr)
      node = {base_attr: {}} #type=dict of dict
      splitSample = sample[base_attr]
      allDistAttr = splitSample.unique()
      next_attr = [attr for attr in allDistAttr if attr != base_attr]

      for attr_value in allDistAttr:
        ssIndex = (splitSample == attr_value)
        sub_sample = sample.iloc[np.where(ssIndex)]
        sub_sample_value = sample_value.iloc[np.where(ssIndex)]
        if len(sub_sample) == 0:
          node[base_attr][attr_value] = sample_value.value_counts().idxmax()
        else:
          node[base_attr][attr_value] = self.DIDO(sub_sample, sub_sample_value, next_attr, depth + 1, base_attr) #calling recursion

      return node
      
    
    # for Discrete input Real output case;
    def DIRO(self, sample_weights, sample: pd.DataFrame, sample_value: pd.Series, next_attr, depth=0, base_attr=None):
      if len(sample_value.drop_duplicates()) == 1 or depth >= self.maxDepth:
      #if there is a single value in dataframe or a single type of data outputs or the max depth has been reached 
      #then it will return the mean of the output column
        return sample_value.mean()

      # if we have reached the end of the tree i.e. incase of no more attributes to split on
      elif len(next_attr) == 0  or len(sample_value) == 0: 
        return base_attr

      base_attr = self.computeRoot(sample, sample_value, next_attr)
      node = {base_attr: {}}
      splitSample = sample[base_attr]
      allDistAttr = splitSample.unique()
      next_attr = [attr for attr in allDistAttr if attr != base_attr]
      
      for attr_value in allDistAttr:
        ssIndex = (splitSample == attr_value)
        sub_sample = sample.iloc[np.where(ssIndex)]
        sub_sample_value = sample_value.iloc[np.where(ssIndex)]
        if len(sub_sample) == 0:
          node[base_attr][attr_value] = sample_value.value_counts().idxmax()
        else:
          node[base_attr][attr_value] = self.DIRO(sub_sample, sub_sample_value, next_attr, depth + 1, base_attr) #calling recursion

      return node


    # getting input for real input 
    def computeRootRI(self, sample : pd.DataFrame, sample_value: pd.Series, sample_weights):
        
            ent = self.criterion(sample_value, sample_weights)
            gains = []
            for col in sample.columns:
                sortedColumn = sample[col].copy().sort_values(ascending= True)
                indices = sortedColumn.index
                newTargets = sample_value.loc[indices]
                #resetting the indicides to the original so that accessiing the 
                #output values will be in sync and not of other indices
                newTargets.reset_index(drop=True, inplace=True) 
                
                for split in range(0, len(sample)):
                    leftRange = newTargets.iloc[:split+1]
                    left_weights = sample_weights[:split+1]
                    rightRange = newTargets.iloc[split+1:] 
                    right_weights = sample_weights[split+1:]

                    entLeft = self.criterion(leftRange, left_weights) * (len(leftRange) / len(sample_value))
                    entRight = self.criterion(rightRange, right_weights) * (len(rightRange) / len(sample_value))

                    gain = ent - (entLeft + entRight)

                    gains.append([gain, col, split])

            highestGain, best_split, splitInd = sorted(gains, key=lambda x: x[0], reverse=True) [0] #the highest gain will be on top of this list
            lim = (sample[best_split].iloc[splitInd] + sample[best_split].iloc[splitInd - 1])/2. #as in real input we will predict Y or N based on
             #inequalities thus limits the threshold value for the split

            return best_split, lim

    # for Real input Discrete output case;
    def RIDO(self, sample: pd.DataFrame, sample_value: pd.Series, sample_weights : pd.Series,  depth=0, base_attr=None):
           
            if depth >= self.maxDepth: 
              #print(sample_value.mode())
              return sample_value.mode().astype(int)

            base_attr, lim = self.computeRootRI(sample, sample_value, sample_weights)

            less_than = sample[base_attr] <= lim
            less_sample = sample.iloc[np.where(less_than)]
            less_sample_value = sample_value.iloc[np.where(less_than)]

            more_than = sample[base_attr] > lim
            more_sample = sample.iloc[np.where(more_than)]
            more_sample_value = sample_value.iloc[np.where(more_than)]


            if len(less_sample_value) == 0:
                return more_sample_value.value_counts().idxmax()
            elif len(more_sample_value) == 0:
                return less_sample_value.value_counts().idxmax()


            node = {f'{base_attr},{lim}' :{ #f'{base_attr} <= {lim}':
                    'Y': self.RIDO(less_sample, less_sample_value, sample_weights, depth + 1, base_attr), 
                    'N': self.RIDO(more_sample, more_sample_value, sample_weights, depth + 1, base_attr)
                }
            }
            return node
    
    # for Real input real output case;
    def RIRO(self, sample: pd.DataFrame, sample_value: pd.Series, depth=0, base_attr=None):

            if depth >= self.maxDepth:                                 # if len(np.unique(sample)) == 1 or depth >= self.maxDepth:
                return sample_value.mean()

            base_attr, lim = self.computeRootRI(sample, sample_value)

            less_than = sample[base_attr] <= lim
            less_sample = sample.iloc[np.where(less_than)]
            less_sample_value = sample_value.iloc[np.where(less_than)]

            more_than = sample[base_attr] > lim
            more_sample = sample.iloc[np.where(more_than)]
            more_sample_value = sample_value.iloc[np.where(more_than)]


            if len(less_sample_value) == 0:
                return more_sample_value.mean()
            elif len(more_sample_value) == 0:
                return less_sample_value.mean()


            node = {
                f'{base_attr},{lim}' :{ #f'{rootAttr} <= {thresh}':
                    'Y': self.RIRO(less_sample, less_sample_value, depth + 1, base_attr), 
                    'N': self.RIRO(more_sample, more_sample_value, depth + 1, base_attr)
                }
            }
            return node

    def fit(self, X, y, sample_weights,depth=0):
            """
            Function to train and construct the decision tree
            it fits the decision tree model to a given dataset (X, y).
            The function takes in two parameters: X (a DataFrame of input examples) and y 
            (a Series of target values).
            """
            if X.dtypes[0].name != 'category':
                if y.dtype.name != 'category':
                    self.criterion = np.var
                    self.tree = self.RIRO(X, y)
                    self.type = 'RIRO'
                else:
                    self.tree = self.RIDO(X, y,sample_weights, depth)
                    self.type = 'RIDO'

            elif X.dtypes[0].name == 'category':
                if y.dtype.name != 'category':
                    self.criterion = np.var
                    self.tree = self.DIRO(X, y, X.columns) #X.colums gives us the 
                    # colums of the Xdataframe and thus making the it return a nested 
                    # dictonary in accordance with RIRO RIDO where in the return type 
                    # is nested dictonary.
                    self.type = 'DIRO'
                else:
                    self.tree = self.DIDO(X, y, X.columns)
                    self.type = 'DIDO'

    def predictDiscrete(self, X):
            """
              Funtion to run the decision tree on test inputs
            """
            pred_disc = []
            for i in range(len(X)):
                features = X.iloc[i]

                tree = self.tree.copy()
                while type(tree) == dict: #tree must be of dict type
                    root_attr = list(tree.keys())[0]
                    attr_value = features[root_attr]
                    tree = tree[root_attr][attr_value]

                pred_disc.append(tree.values())

            return pd.Series(pred_disc)


    def predictReal(self, X): 
            # X = pd.Series(X)
            X = pd.DataFrame(X)
            pred_real = []
            for i in range(len(X)):
                features = X.iloc[i]

                tree = self.tree.copy()
                while type(tree) == dict:
                    
                    key = list(tree.keys())[0]
                    root_attr, lim = key.split(',')
                    root_attr, lim = int(root_attr), float(lim)
                    rootAttrValue = features.loc[root_attr]
                    tree = tree[key]
                    if rootAttrValue <= lim:
                        tree = tree['Y']
                    else:
                        tree = tree['N']

                pred_real.append(tree.values[0])

            #print(pred_real)
            
            return pd.Series(pred_real)
            


    def predict(self,  X: pd.DataFrame):
            """
            Funtion to run the decision tree on a data point
            here the is X pd.DataFrame with rows as samples and columns as features
            and it output pd.Series with rows corresponding to output variable. THe output variable 
            in a row is the prediction for sample in corresponding row in X.
            """

            if self.type in ['DIRO', 'DIDO']:
                return self.predictDiscrete(X)
            else:
                return self.predictReal(X)

                

            
    def plot(self):
            """
            Function to plot the tree
            Output Example:
            ?(X1 > 4)
                Y: ?(X2 > 7)
                    Y: Class A
                    N: Class B
                N: Class C
            Where Y => Yes and N => No
            """      

            if self.type == 'RIDO':
                def plotTree(tree, depth=0):
                    if type(tree) == dict:
                        for root_attr in list(tree.keys()):
                            data = root_attr.replace(',', '<=')
                            string = f'?(X{data})' + '\n' + '\t' * (depth + 1) + 'Y: ' + \
                                plotTree(tree[root_attr]['Y'], depth + 1) + \
                                '\n' + '\t' * (depth + 1) + 'N: ' + \
                                plotTree(tree[root_attr]['N'], depth + 1)
                    else:
                        return f'Class {tree}'
                    return string
                print(plotTree(self.tree))
            elif self.type == 'RIRO':
                def plotTree(tree, depth=0):
                    if type(tree) == dict:
                        for root_attr in list(tree.keys()):
                            data = root_attr.replace(',', '<=')
                            string = f'?(X{data})' + '\n' + '\t' * (depth + 1) + 'Y: ' + \
                                plotTree(tree[root_attr]['Y'], depth + 1) + \
                                '\n' + '\t' * (depth + 1) + 'N: ' + \
                                plotTree(tree[root_attr]['N'], depth + 1)
                    else:
                        return f'{tree}'
                    return string
                print(plotTree(self.tree))

            elif self.type in ['DIRO', 'DIDO']:
                def plotTree(tree, depth=0):
                    if type(tree) == dict:
                        string = ''
                        for root_attr in list(tree.keys()):

                            next_node = tree[root_attr]

                            for vals in list(next_node.keys()):
                                substring = f'?(X{root_attr} == {vals})' + '\n' + '\t' * (depth + 1) #multiplying tabs
                                substring += plotTree(next_node[vals], depth + 1) 
                                string += '\n' + '\t' * (depth) + substring
                    else:
                        return f'{tree}'
                    return string
                print(plotTree(self.tree))
