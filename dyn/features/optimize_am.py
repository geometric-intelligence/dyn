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


def tau_jl(j, n, degree_index):
    """Calculate tau_jl.

    this is the tau matrix. tau = (X^T*X)^-1*X^T

    QUESTION: getting an error here because it is saying that
    the matrix does not have an inverse... because it is
    singular... what should i do??

    POTENTIAL SOLUTION: don't use matrices. just do equivalent
    math to get the desired component, that does not involve
    taking matrix inverses.
    """
    # degree_index +1 so that it will have correct dimensions
    # rows are data points, columns are degrees
    X = np.empty([n, degree_index + 1])

    times = gs.arange(0, n, 1)

    # note: should probably make sure times starts at zero.
    for i_time, time in enumerate(times):
        for degree in range(degree_index + 1):
            X[i_time][degree] = time**degree

    # print(X)
    X_T = X.transpose()

    tau = np.linalg.inv(X_T @ X) @ X_T  # @ is matrix multiplication

    print(tau.shape)

    # in tau, the rows are the degrees, and the n's are the columns.
    # this is different than X.
    return tau[degree_index, j]


def tau_ij(n, degree, i, j, times):
    """Calculate tau_ij.

    tau_ij is the sum of a bunch of tau_jl's.

    variables:
    degree: polynomial degree
    l: the sum over degrees
    """
    tau_jl_sum = 0
    for degree_index in range(degree):
        tau_jl_sum += tau_jl(j, n, degree_index) * times[i] ** degree_index

    return tau_jl_sum


def capital_c(curve, elastic_metric):
    """Compute capital c.

    C is the derivative of the curve divided by the norm of the
    derivativeof the curve.

    this function is modeled off of the f_transform code in geomstats.

    we do calculations in polar coordinates because it is the best way to
    take the exponential of a vector. see:
    https://brilliant.org/wiki/polar-coordinates/

    the "..." in the array splicing means that it can take any shape, but we take
    the last two indices. Therefore, we could pass than one curve to this
    function if we wanted.
    """
    n_sampling_points = curve.shape[-2]
    velocity = (n_sampling_points - 1) * (curve[..., 1:, :] - curve[..., :-1, :])
    # wanted to print to see if only the right column was negative.
    # both columns are a mix of positive and neg.
    # print(velocity)
    polar_velocity = elastic_metric.cartesian_to_polar(velocity)
    # here, is where all the numbers in right column are negative.
    # print(polar_velocity)
    speeds = polar_velocity[..., :, 0]
    # the theta's in polar coordinates. this is what you multiply by to do exponentials.
    args = polar_velocity[..., :, 1]

    c_args = args * elastic_metric.a
    # QUESTION: is this right? want this to be equal to 1. based on gs code.
    c_norms = speeds**0
    c_polar = gs.stack([c_norms, c_args], axis=-1)
    c_cartesian = elastic_metric.polar_to_cartesian(c_polar)

    # these numbers are all fine.
    # print(c_cartesian)

    return c_cartesian


def dr_da(i, curve_trajectory, elastic_metric, n, degree):
    """Calculate the derivative of r_i w.r.t. a.

    Utilizes f_transform in discrete_curves to calculate
    (sqrt of the norm of c prime) multiplied by (capital c
    to the power of a). remember that when we initialized
    elastic_metric, we set b= 0.5.

    returns:
    dr_da: the derivative of r_i w.r.t. a.

    IN PROGRESS:

    PROBLEM: capital c sometimes has negative values. this is
    giving us problems when we compute the log of cap c
    because the log of a negative number is nan. Need
    to figure out how i can get around this.

    note: np.log is equivalent to ln.
    """
    n_times = len(curve_trajectory)
    times = gs.arange(0, n_times, 1)

    # sqrt_norm_c_i_prime = sqrt_norm_c_prime(curve_trajectory[i])
    cap_c_i = capital_c(curve_trajectory[i], elastic_metric)

    j_sum = 0
    for j in range(n):
        # sqrt_norm_c_j_prime = sqrt_norm_c_prime(curve_trajectory[j])
        cap_c_j = capital_c(curve_trajectory[j], elastic_metric)

        #         if j== 0:
        #             #initialize j_sum
        #             j_sum = (
        #                 tau_ij(n, degree, i, j, times)
        #                 * elastic_metric.f_transform(curve_trajectory[j])
        #                 * np..log(cap_c_j)
        #             )
        # we are getting an error because we are trying to take the log of a neg. #.
        # print(cap_c_j)
        # print(np.log(cap_c_j))
        j_sum += tau_ij(n, degree, i, j, times) * np.multiply(
            elastic_metric.f_transform(curve_trajectory[j]), np.log(cap_c_j)
        )
    # print(j_sum.shape)
    #     print(np.multiply(
    #                 elastic_metric.f_transform(curve_trajectory[j]),
    #                 np.log(cap_c_j)
    #             ))

    return (
        np.multiply(elastic_metric.f_transform(curve_trajectory[i]), np.log(cap_c_i))
        - j_sum
    )


def r(i, curve_trajectory, elastic_metric, n, degree):
    """Calculate r_i.

    This function is pretty much the same as dr_da except without the
    log(C)'s.
    """
    n_times = len(curve_trajectory)
    times = gs.arange(0, n_times, 1)

    j_sum = 0
    for j in range(n):

        j_sum += tau_ij(n, degree, i, j, times) * elastic_metric.f_transform(
            curve_trajectory[j]
        )

    return elastic_metric.f_transform(curve_trajectory[i]) - j_sum


def mse_gradient(curve_trajectory, n, n_prime, degree, a):
    """Compute the derivative of the MSE function w.r.t. a.

    put more descriptive caption later.

    essentially, mse = norm2(r_i)
    but r_i = q_i-qhat_i, which are both matrices.
    therefore, to calculate the norm of a matrix, we have to use
    the Frobenius Norm, which essentially sums the squared absolute
    value of every element.
    But, when we take the derivative of mse w.r.t. a, that means
    we are taking the derivative of an absolute value.
    remember that the derivative of an abs value is equal to...

    later, put all the math that i wrote on ipad to give better caption.
    """
    b = 0.5
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    d_mse_sum = 0

    # TO DO: change this so that n_prime times do not start at zero.
    for i in range(n_prime):
        dr_da_var = dr_da(i, curve_trajectory, elastic_metric, n, degree)
        # dr_da_var_trans = np.transpose(dr_da_var)
        r_var = r(i, curve_trajectory, elastic_metric, n, degree)

        rows, cols = dr_da_var.shape
        # print(rows, cols)

        for row in range(rows):
            for col in range(cols):
                # dr_da is returning nan. the others are fine.
                # it also seems that only every other (i.e. 1,3, etc in cols) is
                # bugging.
                # print(dr_da_var[row][col],r_var[row][col],np.abs(r_var[row][col]))
                d_mse_sum += dr_da_var[row][col] * r_var[row][col]
        # print(d_mse_sum)

    return 2 * d_mse_sum


def gradient_descent(
    init_a, learn_rate, max_iter, curve_trajectory, n, n_prime, degree, tol=0.01
):
    """Calculate minimum x using gradient descent.

    structure inspiration source:
    https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21

    sample function also returns steps. we could do that if we want to debug.
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

    return a


def know_m_find_best_a(trajectory, degree):
    """Use a gradient search to find best a, for a given m.

    This function takes a trajectory and a degree and then uses gradient
    descent to find out which value of a minimizes the mean squared error
    (MSE) function.

    IN PROGRESS

    TO DO: calculate mse with given a.
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
    # want to start with degree of 1 becuase that is a line, which is a geodesic
    ms = np.arange(1, 6)
    # best_rmses = np.empty([len(ms)])
    best_as = np.empty([len(ms)])
    # steps = np.empty([len(ms)])
    # min_rmse = 1
    # best_am = np.empty([2])

    for i_m, m in enumerate(ms):
        best_as[i_m] = know_m_find_best_a(trajectory, m)
        print(best_as[i_m])


#         if best_rmses[i_m] < min_rmse:
#             best_am[0] = best_as[i_m]
#             print(best_as[i_m])
#             best_am[1] = m

#     return best_am
