# TODO. implement mape score
import numpy as np


def mape(y_true, y_pred):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
