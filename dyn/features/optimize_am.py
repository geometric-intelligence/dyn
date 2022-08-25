"""Find the optimal a and m for a given trajectory.

a is an elastic metric parameter, and m is the degree of polynomial
regression fitting.

if given a trajectory, this will be able to

will import f_fit_functions
-i will need to create a new function in that that only returns the r^2 value
    without plotting.

Next, i will need to make another file that tests these values against common values
therefore, we will have one file to train, another to validate, another to test.. either
that, or i will include that in this notebook.
"""

import os

import geomstats.backend as gs

# import matplotlib.pyplot as plt
import numpy as np

# import torch

# load discrete curves and R2 manifolds
# from geomstats.geometry.discrete_curves import (
#     R2,
#     ClosedDiscreteCurves,
#     DiscreteCurves,
#     ElasticMetric,
# )

# from geomstats.geometry.euclidean import Euclidean
# from geomstats.geometry.pre_shape import PreShapeSpace
# from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

# import dyn.dyn.features.f_fit_functions as ffit

# import dyn.dyn.features.basic as basic
os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def C(curve):
    """Compute C.

    C is the derivative of the curve divided by the norm of the
    derivativeof the curve.
    """
    # question: how to compute derivative of curve...
    # norm is gs.norm right? --yes

    return curve


def t_squared_bar(times):
    """Calculate the sum of the squared times."""
    return np.sum(np.square(times))


def t_bar(times):
    """Compute t_bar from the times array."""
    return np.sum(times)


def tau(i, j, n):
    """Compute tau for the tau_tilda function."""
    times = gs.arange(0, n, 1)
    if j == 0:
        return (t_bar(times) ** 2 - t_bar(times) * times[i]) / (
            n * (t_squared_bar(times) - t_bar(times) ** 2)
        )
    elif j == 1:
        return -(t_bar(times) + times[i]) / (
            n ** (t_squared_bar(times) - t_bar(times) ** 2)
        )


def tau_tilda(curve, i_index, j_index, n):
    """Compute tau_tilda for the alpha function."""
    if i_index == j_index:
        # question: how to find the derivative of curve in gs
        return 1 - tau(
            i_index, j_index, n
        )  # multiplied by the square root of the norm of the
        # derivative of the curve
    else:
        return -tau(
            i_index, j_index, n
        )  # multiplied by the square root of the norm of the
        # derivative of the curve


def alpha(curve_trajectory, n, n_prime, j, j_prime):
    """Compute alpha for the MSE derivative."""
    alpha = 0
    for i in range(n_prime):
        # question: is log right here?
        alpha += (
            2
            * tau_tilda(curve_trajectory[i], i, j, n)
            * gs.log(C[i])
            * tau_tilda(curve_trajectory[i], j_prime, n)
        )

    return alpha


def mse_gradient(curve_trajectory, n, n_prime, a):
    """Compute the derivative of the MSE function w.r.t. a."""
    d_mse = 0
    for j in range(n):
        for j_prime in range(n):
            d_mse += (
                alpha(curve_trajectory, n, n_prime, j, j_prime)
                * (C[curve_trajectory[j]] * C[curve_trajectory[j_prime]]) ** a
            )

    return d_mse


def gradient_descent(
    init_a, learn_rate, max_iter, curve_trajectory, n, n_prime, tol=0.01
):
    """Calculate minimum x using gradient descent.

    structure inspiration source:
    https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21
    """
    steps = [init_a]  # history tracking
    a = init_a

    for _ in range(max_iter):
        diff = learn_rate * mse_gradient(curve_trajectory, n, n_prime, a)
        if np.abs(diff) < tol:
            break
        a = a - diff
        steps.append(a)  # history tracing

    return steps, a


def know_m_find_best_a(trajectory, degree):
    """Use a gradient search to find best a, for a given m.

    This function takes a trajectory and a degree and then uses gradient
    descent to find out which value of a minimizes the mean squared error
    (MSE) function.

    QUESTION: should we be doing something to transform to curve space again?
    """
    init_a = 0.2
    learn_rate = 1
    max_iter = 100
    n = 10
    n_prime = 5
    tol = 0.01
    if degree == 1:
        return gradient_descent(
            init_a, learn_rate, max_iter, trajectory, n, n_prime, tol
        )
    else:
        print("error: polynomial version not implemented yet")


def find_best_am(trajectory):
    """For a given geodesic, find the (m,a) pair that minimnizes rmse.

    Use a grid search on m and a gradient search on a to find the best pairs of (m,a).
    Then, choose the pair that minimizes root mean squared error.
    """
    ms = np.arange(6)
    best_rmses = np.empty([len(ms)])
    best_as = np.empty([len(ms)])
    min_rmse = 1
    best_am = np.empty([2])

    for i_m, m in enumerate(ms):
        best_as[i_m], best_rmses[i_m] = know_m_find_best_a(trajectory, m)
        print(best_as[i_m])
        if best_rmses[i_m] < min_rmse:
            best_am[0] = best_as[i_m]
            print(best_as[i_m])
            best_am[1] = m

    return best_am
