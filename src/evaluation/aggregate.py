from typing import Optional
import numpy as np

def _identity(X, **kwargs):
    return X

def softmin_pooling(x: np.ndarray, *, temperature: float = 1.0, axis: int = 1) -> np.ndarray:
    """Softmin score function.

    Args:
        x: Input array.
        temperature: Temperature parameter.
        axis: Axis along which to apply the function.

    Returns:
        Softmin score.
    """
    def softmax(scores: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_scores = np.exp(scores / temperature)
        return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)
    
    return np.einsum('ij,ij->i', x, softmax(1 - x))


# Pool scores s = (s_1, ..., s_K) by first transforming them to r_i = w_i * log(s_i + eps) + b_i and apply mean pooling to r.
def log_transform_pooling(s: np.ndarray, *, weights: np.ndarray, biases: np.ndarray, eps: float = 1e-8, axis: int = 1) -> np.ndarray:
    """Log transform score function.

    For a score vector s = (s_1, ..., s_K) with K scores, the function evaluates
    r_i = w_i * log(s_i + eps) + b_i for each i and returns the mean of the
    resulting vector r = (r_1, ..., r_K).
    

    Parameters
    ----------
    s :
        Scores to be transformed.
    weights :
        Weights used to scale the log of the scores.
    biases :
        Biases used to shift the log of the scores.
    eps:
        Small value added to the scores to avoid numerical issues with log(0).

    Returns:
        Log transform score.
    """
    r = weights * np.log(s + eps) + biases
    return np.mean(r, axis=axis)


def cumulative_average(s: np.ndarray, *, k: int = 1, axis: int = 1) -> np.ndarray: 
    """Cumulative average score function.

    For a score vector s = (s_1, ..., s_K) with K scores, the function evaluates
    sorts the scores in ascending order and returns the cumulative average of
    the first k scores.

    Parameters
    ----------
    s :
        Scores to be transformed.
    k :
        Number of scores to be averaged.

    Returns:
        Cumulative average score.
    """
    s_sorted = np.sort(s, axis=axis)
    return np.mean(s_sorted[:, :k], axis=axis)


def simple_moving_average(s: np.ndarray, *, k: int = 1, axis: int = 1) -> np.ndarray: 
    """Simple moving average score function.

    For a score vector s = (s_1, ..., s_K) with K scores, the function evaluates
    sorts the scores in ascending order and returns the mean of the simple moving average
    of all scores: The unweighted mean of the previous k scores.

    Parameters
    ----------
    s :
        Scores to be transformed.
    k :
        Number of scores to be averaged.

    Returns:
        Simple moving average score.
    """


    s_sorted = np.sort(s, axis=axis)
    if k == 1:
        return np.mean(s_sorted, axis=axis)
    
    N = s.shape[0]
    K_ma = s.shape[1] - k + 1
    s_ma = np.zeros((N, K_ma))
    s_ma[:, 0] = np.mean(s_sorted[:, :k], axis=axis)
    for i in range(1,K_ma):
        prev = s_ma[:, i-1]
        next_point = s_sorted[:, i+k-1]
        old_point = s_sorted[:, i-1]
        s_ma[:, i] = prev + (next_point - old_point) / k

    return np.mean(s_ma, axis=axis)

def exponential_moving_average(s: np.ndarray, *, alpha: Optional[float] = None, axis: int = 1) -> np.ndarray: 
    """Exponential moving average (EMA) score function.

    For a score vector s = (s_1, ..., s_K) with K scores, the function evaluates
    sorts the scores in *descending* order and returns the exponential moving average
    of all scores.

    Parameters
    ----------
    s :
        Scores to be transformed.
    alpha :
        Discount factor that determines the weight of the previous EMA score.

    Returns:
        Exponential moving average score.
    """
    s_sorted = np.sort(s, axis=axis)[:, ::-1]
    _, K = s.shape
    if alpha is None:
        # One conventional choice for alpha is 2/(K +1), where K is the number of periods in the moving average.
        alpha = float(2 / (K + 1))
    s_ema = s_sorted[:, 0]
    for i in range(1, K):
        s_ema = alpha * s_sorted[:, i] + (1 - alpha) * s_ema
    return s_ema


def weighted_cumulative_average(s: np.ndarray, *, weights: Optional[np.ndarray] = None, axis: int = 1) -> np.ndarray:
    """Weighted cumulative average score function.

    For a score vector s = (s_1, ..., s_K) with K scores, the function evaluates
    sorts the scores in ascending order and returns the weighted cumulative average
    of all scores.

    Parameters
    ----------
    s :
        Scores to be transformed.
    f :
        Weighting function. Takes the rank (0-based) of the score as input and returns
        the weight for that score.

    Returns:
        Weighted cumulative average score.
    """
    s_sorted = np.sort(s, axis=axis)
    N, K = s.shape
    if weights is None:
        weights = np.exp(-np.arange(K))
    means = np.zeros((N, K))
    for i in range(K):
        means[:, i] = np.mean(s_sorted[:, :(i+1)], axis=axis)
    return np.sum(weights * means, axis=axis)