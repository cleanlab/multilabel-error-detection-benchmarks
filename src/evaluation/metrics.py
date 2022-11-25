import numpy as np
from sklearn.metrics import average_precision_score


def lift_at_k(y_true: np.array, y_score: np.array, k: int = 100) -> np.float:
    """Compute Lift at K evaluation metric"""

    # sort scores
    sort_indices = np.argsort(y_score)

    # compute lift for the top k values
    lift_at_k = y_true[sort_indices][-k:].mean() / y_true.mean()

    return lift_at_k


def average_precision_at_k(
    y_true: np.array, y_score: np.array, k: int = 100
) -> np.float:
    """Compute Average Precision at K evaluation metric"""

    # sort scores
    sort_indices = np.argsort(y_score)

    # compute average precision for the top k values
    ap_at_k = average_precision_score(
        y_true[sort_indices][-k:], y_score[sort_indices][-k:]
    )

    return ap_at_k
