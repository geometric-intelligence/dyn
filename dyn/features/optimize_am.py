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
from geomstats.geometry.discrete_curves import R2, ElasticMetric

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


# def c_prime(curve):
#     """Calculate c prime.

#     parameters:
#     -----------
#     velocity: c prime
#     speeds: norm of velocity.

#     note that c prime is just the derivative between points along
#     the curve. it is not the time derivative of the curve along
#     the manifold.
#     """
#     n_sampling_points = curve.shape[-2]
#     velocity = (n_sampling_points - 1) * (curve[..., 1:, :] - curve[..., :-1, :])
#     polar_velocity = self.cartesian_to_polar(velocity)
#     speeds = polar_velocity[..., :, 0]
#     args = polar_velocity[..., :, 1]

#     c_prime_norm = * gs.sqrt(speeds)

#     return c_prime_norm


# def sqrt_norm_c_prime(curve):
#     """Calculate the square root of the norm of c prime.

#     might not need, if we use f transform function...
#     """
#     return np.sqrt(gs.linalg.norm(c_prime(curve)))


def tau_jl(curve, j, degree_index, times):
    """Calculate tau_jl.

    Damn... might need to do some matrix multiplication.
    make a vector of betas.

    then beta[0] = tau_j0
    beta[1] = tau_j1
    etc.

    waiiiittt actually, looking at equation 12.
    maybe better to take f out of dr_da equation becaues beta actually
    includes q_i which is the f transform.

    yes, so just have an entire function dedicated to calculating the
    vector beta

    so my goal is to find dr/da in terms of beta. I want to do this
    becasue i am going to have to do the matrix multiplication anyway in
    order to calculate these tau's.


    ACTUALLY:
    new idea: now, i think we are just going to calculate the tau matrix
    we might be able to combine the tau_ij and the tau_jl functions into
    one tau matrix.

    no... ok, so tau_jl is the tau matrix ((X^T)X)^-1*X^T
    so we just have to calculate that, which essentially means
    we just have to calculate the X matrix, its transverse,
    and the inverse of those two matrices multiplied.
    """


def tau_ij(curve, degree, i, j, times):
    """Calculate tau_ij.

    tau_ij is the sum of a bunch of tau_jl's.

    variables:
    degree: polynomial degree
    l: the sum over degrees
    """
    tau_jl_sum = 0
    for degree_index in range(degree):
        tau_jl_sum += tau_jl(curve, j, degree_index, times) * times[i] ** degree_index

    return tau_jl_sum


def capital_c(curve, elastic_metric):
    """Compute capital c.

    C is the derivative of the curve divided by the norm of the
    derivativeof the curve.
    """
    # question: how to compute derivative of curve...
    # norm is gs.norm right? --yes
    # we compute the derivative by taking difference
    # between sampling points. example in f transform
    # in geomstats

    n_sampling_points = curve.shape[-2]
    velocity = (n_sampling_points - 1) * (curve[..., 1:, :] - curve[..., :-1, :])
    polar_velocity = elastic_metric.cartesian_to_polar(velocity)
    # QUESTION: what are these dots?
    # speeds = polar_velocity[..., :, 0]
    args = polar_velocity[..., :, 1]

    # QUESTION: why do all these calculations in polar?
    capital_c_polar = args * elastic_metric.a
    # c_prime_norm_sqrt = gs.sqrt(speeds)
    capital_c_cartesian = elastic_metric.polar_to_cartesian(capital_c_polar)

    #     QUESTION: why is args multiplied by the exponent instead of not...
    #     f_args = args * self.a / (2 * self.b)
    #     f_norms = 2 * self.b * gs.sqrt(speeds)
    # QUESTION: what is this stack thing?
    #     f_polar = gs.stack([f_norms, f_args], axis=-1)
    #     f_cartesian = self.polar_to_cartesian(f_polar)

    return capital_c_cartesian


def dr_da(i, curve_trajectory, elastic_metric, n, degree):
    """Calculate the derivative of r_i w.r.t. a.

    Utilizes f_transform in discrete_curves to calculate
    (sqrt of the norm of c prime) multiplied by (capital c
    to the power of a). remember that when we initialized
    elastic_metric, we set b= 0.5.

    returns:
    dr_da: the derivative of r_i w.r.t. a.
    """
    n_times = len(curve_trajectory)
    times = gs.arange(0, n_times, 1)

    # sqrt_norm_c_i_prime = sqrt_norm_c_prime(curve_trajectory[i])
    cap_c_i = capital_c(curve_trajectory[i], elastic_metric)

    j_sum = 0
    for j in range(n):
        # sqrt_norm_c_j_prime = sqrt_norm_c_prime(curve_trajectory[j])
        cap_c_j = capital_c(curve_trajectory[j], elastic_metric)

        j_sum += (
            tau_ij(curve_trajectory[j], degree, i, j, times)
            * elastic_metric.f_transform(curve_trajectory[j])
            * gs.log(cap_c_j)
        )

    return elastic_metric.f_transform(curve_trajectory[i]) * gs.log(cap_c_i) - j_sum


def r(i, curve_trajectory, a, n, degree):
    """Calculate r_i.

    IN PROGRESS.

    need to change inputs to match dr_da.

    This function is pretty much the same as dr_da except without the
    log(C)'s.
    """
    n_times = len(curve_trajectory)
    times = gs.arange(0, n_times, 1)

    return times


def mse_gradient(curve_trajectory, n, n_prime, degree, a):
    """Compute the derivative of the MSE function w.r.t. a.

    QUESTION: do dr_da and r need to be dotted? not
    multiplied?
    """
    b = 0.5
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    d_mse_sum = 0

    for i in range(n_prime):
        d_mse_sum += dr_da(i, curve_trajectory, elastic_metric, n, degree) * r(
            i, curve_trajectory, elastic_metric, n, degree
        )

    return 2 * d_mse_sum


def gradient_descent(
    init_a, learn_rate, max_iter, curve_trajectory, n, n_prime, degree, tol=0.01
):
    """Calculate minimum x using gradient descent.

    structure inspiration source:
    https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21
    """
    steps = [init_a]  # history tracking
    a = init_a

    for _ in range(max_iter):
        # gradient must be a function of a.
        diff = learn_rate * mse_gradient(curve_trajectory, n, n_prime, degree, a)
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
    return gradient_descent(
        init_a, learn_rate, max_iter, trajectory, n, n_prime, degree, tol
    )


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


# def t_squared_bar(times):
#     """Calculate the sum of the squared times."""
#     return np.sum(np.square(times))


# def t_bar(times):
#     """Compute t_bar from the times array."""
#     return np.sum(times)


# def tau(i, j, n):
#     """Compute tau for the tau_tilda function."""
#     times = gs.arange(0, n, 1)
#     if j == 0:
#         return (t_bar(times) ** 2 - t_bar(times) * times[i]) / (
#             n * (t_squared_bar(times) - t_bar(times) ** 2)
#         )
#     elif j == 1:
#         return -(t_bar(times) + times[i]) / (
#             n ** (t_squared_bar(times) - t_bar(times) ** 2)
#         )


# def tau_tilda(curve, i_index, j_index, n):
#     """Compute tau_tilda for the alpha function."""
#     if i_index == j_index:
#         # question: how to find the derivative of curve in gs
#         return 1 - tau(
#             i_index, j_index, n
#         )  # multiplied by the square root of the norm of the
#         # derivative of the curve
#     else:
#         return -tau(
#             i_index, j_index, n
#         )  # multiplied by the square root of the norm of the
#         # derivative of the curve


# def alpha(curve_trajectory, n, n_prime, j, j_prime):
#     """Compute alpha for the MSE derivative."""
#     alpha = 0
#     for i in range(n_prime):
#         # question: is log right here?
#         alpha += (
#             2
#             * tau_tilda(curve_trajectory[i], i, j, n)
#             * gs.log(C[i])
#             * tau_tilda(curve_trajectory[i], j_prime, n)
#         )

#     return alpha


# def mse_gradient(curve_trajectory, n, n_prime, a):
#     """Compute the derivative of the MSE function w.r.t. a."""
#     d_mse = 0
#     for j in range(n):
#         for j_prime in range(n):
#             d_mse += (
#                 alpha(curve_trajectory, n, n_prime, j, j_prime)
#                 * (C[curve_trajectory[j]] * C[curve_trajectory[j_prime]]) ** a
#             )

#     return d_mse
