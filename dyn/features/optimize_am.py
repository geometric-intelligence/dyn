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


def tau_jl(j, n, degree_index, times):
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

    to save computation, could probably just make this a function
    that returns the tau matrix, calculate once, save it, and
    then call on it in the for loop
    """
    X = np.empty([degree_index, n])

    # unnecessary because (anything)^0 is 1.
    # X[:,0]=1

    # note: should probably make sure times starts at zero.
    for i_time, time in enumerate(times):
        for i_degree, degree in enumerate(degree_index):
            X[i_time, i_degree] = time**degree

    return X


def tau_ij(n, degree, i, j, times):
    """Calculate tau_ij.

    tau_ij is the sum of a bunch of tau_jl's.

    variables:
    degree: polynomial degree
    l: the sum over degrees
    """
    tau_jl_sum = 0
    for degree_index in range(degree):
        tau_jl_sum += tau_jl(j, n, degree_index, times) * times[i] ** degree_index

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
    polar_velocity = elastic_metric.cartesian_to_polar(velocity)
    speeds = polar_velocity[..., :, 0]
    # the theta's in polar coordinates. this is what you multiply by to do exponentials.
    args = polar_velocity[..., :, 1]

    c_args = args * elastic_metric.a
    # QUESTION: is this right? want this to be equal to 1. based on gs code.
    c_norms = speeds**0
    c_polar = gs.stack([c_norms, c_args], axis=-1)
    c_cartesian = elastic_metric.polar_to_cartesian(c_polar)

    return c_cartesian


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
            tau_ij(n, degree, i, j, times)
            * elastic_metric.f_transform(curve_trajectory[j])
            * gs.log(cap_c_j)
        )

    # this is a 2D array. perhaps this means that we actually did not do the dotting
    # correctly in the mse derivative function.
    # print((elastic_metric.f_transform(curve_trajectory[i])
    # * gs.log(cap_c_i) - j_sum).shape)

    return elastic_metric.f_transform(curve_trajectory[i]) * gs.log(cap_c_i) - j_sum


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

    IN PROGRESS.

    still need to dot the dr_da and r components.

    remember that r is the difference between q_i real and q_i predicted.
    q's are still curves, just in linear space. q_i is one curve.
    therefore, q_i - q_i_predicted is the difference between curves.

    why are we doing subtraction stuff instead of finding the distance
    between two curves on the manifold of discrete curves? shouldnt we be
    comparing curves on that manifold? how does finding the difference between
    each of the points on the curves give us useful information?

    However, if we do compare them in linear space, I'm guessing we would
    want to dot all the rows of dr_da with the rows of r. That would give us
    a 1D vector... which is not a scalar, like it should be.

    I believe i must be having problems because maybe i don't know how
    exactly to find the norm a 2D array. I only know how to find the norm
    of a 1D array.


    NOTE: we may actually have to re-derive things. i looked at the
    numpy source code for getting the norm of a matrix, and they use
    the Frobenius Norm. Since we are not dealing with vectors, i don't know if
    it is valid to use our current formula to get the norm of them.
    https://mathworld.wolfram.com/FrobeniusNorm.html

    ALTERNATIVELY: there is another source that says that the norm
    of a 2D vector is the same as the norm of that 2D vector, reshaped.
    https://www.geeksforgeeks.org/find-a-matrix-or-vector-norm-using-numpy/
    """
    b = 0.5
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    d_mse_sum = 0

    for i in range(n_prime):
        dr_da_var = dr_da(i, curve_trajectory, elastic_metric, n, degree)
        # dr_da_var_trans = np.transpose(dr_da_var)
        r_var = r(i, curve_trajectory, elastic_metric, n, degree)

        d_mse_sum += dr_da_var * r_var
        print(dr_da_var.shape)
        print(r_var.shape)

    print(d_mse_sum.shape)

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
    ms = np.arange(6)
    # best_rmses = np.empty([len(ms)])
    best_as = np.empty([len(ms)])
    steps = np.empty([len(ms)])
    # min_rmse = 1
    # best_am = np.empty([2])

    for i_m, m in enumerate(ms):
        steps[i_m], best_as[i_m] = know_m_find_best_a(trajectory, m)
        print(best_as[i_m])


#         if best_rmses[i_m] < min_rmse:
#             best_am[0] = best_as[i_m]
#             print(best_as[i_m])
#             best_am[1] = m

#     return best_am
