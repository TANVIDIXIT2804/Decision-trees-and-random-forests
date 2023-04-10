import pandas as pd
from typing import Union
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    total = len(y)
    correct=sum(y_hat == y)
    return correct/total
    

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    pred_pos = y_hat == cls
    if sum(pred_pos) > 0:
        return (y_hat[pred_pos] == y[pred_pos]).sum()/pred_pos.sum()
    else:
        return None
    
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert(y_hat.size == y.size)
    assert(y_hat.size > 0)
    act_pos = y == cls
    if sum(act_pos) > 0:
        return (y_hat[act_pos] == y[act_pos]).sum()/act_pos.sum()
    else:
        return None
    pass

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    pr = np.array(y_hat)
    gt = np.array(y)

    er = (pr-gt)**2
    return(np.sqrt(np.mean(er)))
    pass

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    pr = np.array(y_hat)
    gt = np.array(y)
    er = pr - gt
    return(np.mean(abs(er)))
    pass
