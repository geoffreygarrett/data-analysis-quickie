""" helper_functions.py


"""

# Authorship ----------------------------------------------------------------------------------------------------------#
__author__      = "Geoffrey Hyde Garrett"
__copyright__   = None
__credits__     = None
__license__     = "MIT"
__version__     = "1.0.0"
__maintainer__  = "Geoffrey Hyde Garrett"
__email__       = "g.h.garrett13@gmail.com"
__status__      = "Pre-alpha"

# Imports -------------------------------------------------------------------------------------------------------------#
import numpy as np


def parameter_covariance(H, Pyy=None):
    if Pyy is not None:
        return np.linalg.inv(np.matmul(H.T, np.matmul(np.linalg.inv(Pyy), H)))
    else:
        return np.linalg.inv(np.matmul(H.T, H))


def unweighted_least_squares(H, y):
    """
    :param H: Information matrix (np.ndarray)
    :param y: Vector of observations (np.ndarray)
    :return: Vector of parameters (np.ndarray)
    """
    # m > n
    if H.shape[0] > H.shape[1]:
        return np.matmul(np.linalg.inv(np.matmul(H.T, H)), np.matmul(H.T, y))
    # m = n
    elif H.shape[0] == H.shape[1]:
        return np.matmul(np.linalg.inv(H), y)
    # m < n
    elif H.shape[0] < H.shape[1]:
        return np.matmul(H.T, np.matmul(np.linalg.inv(np.matmul(H, H.T)), y))


def weighted_least_squares(H, y, Py):
    """
    TODO: (**) Complete other forms with covariance !=(m>n).
    :param H:
    :param y:
    :param Py:
    :return:
    """
    # m > n
    if H.shape[0] > H.shape[1]:
        return np.matmul(np.linalg.inv(np.matmul(np.matmul(H.T, np.linalg.inv(Py)), H)),
                         np.matmul(np.matmul(H.T, np.linalg.inv(Py)), y))
    else:
        raise NotImplementedError("TODO: Complete other forms with covariance !=(m>n)")


def kth_order_moment_about_zero(x, k, bins=None):
    """
    TODO: Complete kth_order_moment_about_zero doc-strings.
    :param x:
    :param k:
    :param bins:
    :return:
    """
    if bins:
        p_i, edges = np.histogram(x, density=True, bins=30)
        x_i = [np.mean([edges[i], edges[i + 1]]) for i in range(len(edges) - 1)]
        return np.sum(np.multiply(np.power(x_i, k), p_i))
    else:
        return np.sum(np.multiply(np.power(x, k), x / np.sum(x)))


def kth_order_moment_about_mean(x, k, bins=None):
    """
    TODO: Complete kth_order_moment_about_mean doc-strings.
    :param x:
    :param k:
    :param bins:
    :return:
    """
    if bins:
        p_i, edges = np.histogram(x, density=True, bins=30)
        x_i = [np.mean([edges[i], edges[i + 1]]) for i in range(len(edges) - 1)]
        return np.sum(np.multiply(np.power(x_i - np.mean(x), k), p_i))
    else:
        return np.sum(np.multiply(np.power(x - np.mean(x), k), x / np.sum(x)))

