# TODO. implement adjusted r2 score
import numpy as np
from metrics.r2_score import r2_score


def adjusted_r2(y_true, y_pred, n_features):
   
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    p = n_features

    r2 = r2_score(y_true, y_pred)

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
