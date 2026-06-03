# TODO. implement mae score
import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()


    return np.mean(np.abs(y_true - y_pred))
