
import pandas as pd
import numpy as np

def entropy(Y: pd.Series, sample_weights: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    entropy = 0
    # err = 1e-100
    # p= (Y.value_counts()+err)/(len(Y)+err)
    # print(p.values)
    #print(len(p)) 
    # print(sample_weights.shape)
    # print(Y.unique().shape) 
    Y = pd.Series(Y)
    sample_weights = pd.Series(sample_weights)
    # print('Y',Y)
    # print('sample_weights',sample_weights)
    Y = Y.reset_index(drop = True)
    sample_weights = sample_weights.reset_index(drop = True)
    # print('Y',Y)
    # print('sample_weights',sample_weights)
    df = pd.DataFrame([Y,sample_weights], columns = ['Y','sample_weights'])
    # print(df)
    for cls in Y.unique():
      p = df[df['sample_weights'] == df['Y']]['sample_weights'].sum()/df['sample_weights'].sum()
      entropy = -(np.dot(p,np.log2(p)))
      #print(entropy)
    return entropy
      


def gini_index(Y: pd.Series, sample_weights: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    gi = 0
    # err = 1e-100
    # if len(Y)!=0:
    #   p = (Y.value_counts())/(len(Y)) 
    for cls in Y.unique():
        p = sample_weights[Y==cls].sum()/sample_weights.sum()
        gi = 1 - np.dot(p,p)
    return gi
    


def information_gain(Y: pd.Series, attr: pd.Series, sample_weights: pd.Series, eval = entropy) -> float:
    """
    Function to calculate the information gain
    """
    info_gain = eval(Y)
    data = pd.DataFrame({'attr':attr, 'Y':Y, 'sample_weights':sample_weights})

    for a in set(attr):
      ss = data[data['attr']==a]['Y']
      if len(Y)!=0:
        info_gain = info_gain - ((len(ss))/len(Y))*eval(ss)
        for cls in Y.unique():
          p = sample_weights[Y==cls].sum()/sample_weights.sum()
      else:
        pass

    return info_gain
