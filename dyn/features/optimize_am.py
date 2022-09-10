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


def tau_jl(j_train, times_train, degree_index):
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
    X = np.empty([len(times_train), degree_index + 1])

    # note: should probably make sure times starts at zero.
    for i_time, time in enumerate(times_train):
        for degree in range(degree_index + 1):
            X[i_time][degree] = time**degree

    # print(X)
    X_T = X.transpose()

    tau = np.linalg.inv(X_T @ X) @ X_T  # @ is matrix multiplication

    # in tau, the rows are the degrees, and the n's are the columns.
    # this is different than X.
    return tau[degree_index, j_train]


def tau_ij(times_train, degree, i_val, j_train, times):
    """Calculate tau_ij.

    tau_ij is the sum of a bunch of tau_jl's.

    variables:
    degree: polynomial degree
    l: the sum over degrees
    """
    tau_jl_sum = 0
    for degree_index in range(degree):
        tau_jl_sum += (
            tau_jl(j_train, times_train, degree_index) * times[i_val] ** degree_index
        )

    return tau_jl_sum


def derivative_q_curve(curve, elastic_metric):
    """Compute the derivative of a curve in q-space w.r.t. a.

    we are taking the derivative of:
    norm(curve)^(1/2)*[curve/norm(curve)]^a w.r.t a.

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
    args = polar_velocity[..., :, 1]

    der_args = args * elastic_metric.a + np.pi / 2
    der_norms = args * gs.sqrt(speeds)
    der_polar = gs.stack([der_norms, der_args], axis=-1)
    der_cartesian = elastic_metric.polar_to_cartesian(der_polar)

    return der_cartesian


# def capital_c(curve, elastic_metric):
#     """Compute capital c.

#     C is the derivative of the curve divided by the norm of the
#     derivativeof the curve.

#     this function is modeled off of the f_transform code in geomstats.

#     we do calculations in polar coordinates because it is the best way to
#     take the exponential of a vector. see:
#     https://brilliant.org/wiki/polar-coordinates/

#     the "..." in the array splicing means that it can take any shape, but we take
#     the last two indices. Therefore, we could pass than one curve to this
#     function if we wanted.
#     """
#     n_sampling_points = curve.shape[-2]
#     velocity = (n_sampling_points - 1) * (curve[..., 1:, :] - curve[..., :-1, :])
#     # wanted to print to see if only the right column was negative.
#     # both columns are a mix of positive and neg.
#     # print(velocity)
#     polar_velocity = elastic_metric.cartesian_to_polar(velocity)
#     # here, is where all the numbers in right column are negative.
#     # print(polar_velocity)
#     speeds = polar_velocity[..., :, 0]
#     # the theta's in polar coordinates. this is what you multiply by to do
#  exponentials.
#     args = polar_velocity[..., :, 1]

#     c_args = args * elastic_metric.a
#     # QUESTION: is this right? want this to be equal to 1. based on gs code.
#     c_norms = speeds**0
#     c_polar = gs.stack([c_norms, c_args], axis=-1)
#     c_cartesian = elastic_metric.polar_to_cartesian(c_polar)

#     # these numbers are all fine.
#     # print(c_cartesian)

#     return c_cartesian


def dr_mse_da(i_val, curve_trajectory, elastic_metric, times_train, degree):
    """Calculate the derivative of r_mse for a given i_val w.r.t. a.

    Utilizes f_transform in discrete_curves to calculate
    (sqrt of the norm of c prime) multiplied by (capital c
    to the power of a). remember that when we initialized
    elastic_metric, we set b= 0.5.

    returns:
    dr_da: the derivative of r_mse w.r.t. a.
    """
    n_times = len(curve_trajectory)
    times = gs.arange(0, n_times, 1)

    fit_sum = 0
    for j_train, time in enumerate(times_train):

        fit_sum += tau_ij(
            times_train, degree, i_val, j_train, times
        ) * derivative_q_curve(curve_trajectory[j_train], elastic_metric)

    return derivative_q_curve(curve_trajectory[i_val], elastic_metric) - fit_sum


def r_mse(i_val, curve_trajectory, elastic_metric, times_train, degree):
    """Calculate r_mse for a given i_val (i.e. a given s parameter).

    This function is pretty much the same as dr_da except without the
    log(C)'s.
    """
    n_times = len(curve_trajectory)
    times = gs.arange(0, n_times, 1)

    fit_sum = 0
    for j_train, time in enumerate(times_train):

        fit_sum += tau_ij(
            times_train, degree, i_val, j_train, times
        ) * elastic_metric.f_transform(curve_trajectory[j_train])

    return elastic_metric.f_transform(curve_trajectory[i_val]) - fit_sum


def d_mse(curve_trajectory, elastic_metric, times_train, times_val, degree, a):
    """Compute the derivative of the MSE function w.r.t. a.

    put more descriptive caption later.


    will have to change this part...

    essentially, mse = norm2(r_i)
    but r_i = q_i-qhat_i, which are both matrices.
    therefore, to calculate the norm of a matrix, we have to use
    the Frobenius Norm, which essentially sums the squared absolute
    value of every element.
    But, when we take the derivative of mse w.r.t. a, that means
    we are taking the derivative of an absolute value.
    remember that the derivative of an abs value is equal to...

    later, put all the math that i wrote on ipad to give better caption.

    ... end of part i need to change

    we must multiply each element by (1/the number of sampling points -1)
    because we are
    performing a riemannian integral sum of the norm of r_i(s)'s.
    where i is the sampling point, and s is the parameter that
    parametrizes all curves and ranges from 0 to 1.
    the number of sampling points reduces by 1 because we are operating
    in q space.

    parameters:
    -----------
    r: is the distance between a q curve and the q curve expected from the
        regression fit
    dr_da: the derivative of the distance between a q curve and the q curve
        expected from the regression fit
    """
    n_sampling_points = len(curve_trajectory[0][:, 0])

    d_mse_sum = 0

    # TO DO: change this so that n_prime times do not start at zero.
    for i_val, time in enumerate(times_val):
        dr_da = dr_mse_da(i_val, curve_trajectory, elastic_metric, times_train, degree)
        r = r_mse(i_val, curve_trajectory, elastic_metric, times_train, degree)

        rows, cols = dr_da.shape

        for row in range(rows):
            for col in range(cols):

                d_mse_sum += dr_da[row][col] * r[row][col] / (n_sampling_points - 1)

    return 2 * d_mse_sum


def mse(curve_trajectory, elastic_metric, times_train, times_val, degree, a):
    """Compute the mean squared error (MSE) with the given parameters.

    Computes the sum of the distances between each datapoint and its
    corresponding predicted datapoint.

    parameters:
    -----------
    r: is the distance between a q curve and the q curve expected from the
        regression fit
    dr_da: the derivative of the distance between a q curve and the q curve
        expected from the regression fit
    """
    n_sampling_points = len(curve_trajectory[0][:, 0])

    mse_sum = 0

    # TO DO: change this so that n_prime times do not start at zero.
    # however, might get other indexing errors from this
    for i_val, time in enumerate(times_val):
        r = r_mse(i_val, curve_trajectory, elastic_metric, times_train, degree)

        rows, cols = r.shape

        for row in range(rows):
            for col in range(cols):
                mse_sum += r[row][col] * r[row][col] / (n_sampling_points - 1)

    return mse_sum


def r_var(curve, elastic_metric, q_mean):
    """Compute the distance between the given q curve and the mean q curve."""
    return elastic_metric.f_transform(curve) - q_mean


def dr_var_da(curve, elastic_metric, d_q_mean):
    """Compute the derivative of the distance between the q curve and mean curve."""
    return derivative_q_curve(curve, elastic_metric) - d_q_mean


def var(curve_trajectory, elastic_metric, times_val, a):
    """Compute the total variation of the validation dataset.

    computes sum_i_train(norm(q_curve_i - mean_q_curve))

    parameters:
    -----------
    r: is the distance between a q curve and the mean q curve
    dr_da: the derivative of the distance between a q curve and the mean q curve.
    """
    n_sampling_points = len(curve_trajectory[0][:, 0])

    q_mean = 0
    for i_val, time in enumerate(times_val):
        q_mean += elastic_metric.f_transform(curve_trajectory[i_val])

    q_mean = q_mean / len(times_val)

    var_sum = 0
    for i_val, time in enumerate(times_val):
        r = r_var(curve_trajectory[i_val], elastic_metric, q_mean)

        rows, cols = r.shape

        for row in range(rows):
            for col in range(cols):
                var_sum += r[row][col] * r[row][col] / (n_sampling_points - 1)

    return var_sum


def d_var(curve_trajectory, elastic_metric, times_val, a):
    """Compute the total variation of the validation dataset w.r.t. a.

    computes derivative of sum_i_train(norm(q_curve_i - mean_q_curve))

    parameters:
    -----------
    r: is the distance between a q curve and the mean q curve
    dr_da: the derivative of the distance between a q curve and the mean q curve.
    """
    n_sampling_points = len(curve_trajectory[0][:, 0])

    derivative_q_mean = 0
    for i_val, time in enumerate(times_val):
        derivative_q_mean += derivative_q_curve(curve_trajectory[i_val], elastic_metric)
    derivative_q_mean = derivative_q_mean / len(times_val)

    q_mean = 0
    for i_val, time in enumerate(times_val):
        q_mean += elastic_metric.f_transform(curve_trajectory[i_val])
    q_mean = q_mean / len(times_val)

    d_var_sum = 0
    for i_val, time in enumerate(times_val):
        dr_da = dr_var_da(curve_trajectory[i_val], elastic_metric, derivative_q_mean)
        r = r_var(curve_trajectory[i_val], elastic_metric, q_mean)

        rows, cols = dr_da.shape

        for row in range(rows):
            for col in range(cols):

                d_var_sum += dr_da[row][col] * r[row][col] / (n_sampling_points - 1)

    return 2 * d_var_sum


def r_squared_gradient(curve_trajectory, times_train, times_val, degree, a):
    """Compute the derivative of the r^2 function w.r.t. a.

    We are using the r^2 function as our "loss" function.
    i.e. we are using the r^2 function to evaluate how well our
    value of a and degree choice is working.

    r^2 = (sum of fit variation)/(total variation in dataset)

    r^2 = sum_i^n_val(norm(q_i - q_hat_i)^2) -
            sum_i_n_val(norm(q_i - q_mean)^2)
    """
    b = 0.5
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    fit_variation = mse(
        curve_trajectory, elastic_metric, times_train, times_val, degree, a
    )
    d_fit_variation = d_mse(
        curve_trajectory, elastic_metric, times_train, times_val, degree, a
    )
    total_variation = var(curve_trajectory, elastic_metric, times_val, a)
    d_total_variation = d_var(curve_trajectory, elastic_metric, times_val, a)

    gradient = (
        d_fit_variation * total_variation + fit_variation * d_total_variation
    ) / total_variation**2

    return gradient


def gradient_descent(
    init_a,
    learn_rate,
    max_iter,
    curve_trajectory,
    times_train,
    times_val,
    degree,
    tol=0.01,
):
    """Calculate minimum x using gradient descent.

    structure inspiration source:
    https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21

    sample function also returns steps. we could do that if we want to debug.
    """
    steps = [init_a]  # history tracking
    a = init_a

    for _ in range(max_iter):
        if a >= 0:
            # gradient must be a function of a.

            diff = learn_rate * r_squared_gradient(
                curve_trajectory, times_train, times_val, degree, a
            )
            if np.abs(diff) < tol:
                break
            if a - diff < 0:
                break
            a = a - diff
            steps.append(a)  # history tracing

    return a


# def gradient_ascent(
#     init_a,
#     learn_rate,
#     max_iter,
#     curve_trajectory,
#     times_train,
#     times_val,
#     degree,
#     tol=0.01,
# ):
#     """Calculate maximum value of r2 using gradient ascent.

#     structure inspiration source:
#     https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21

#     sample function also returns steps. we could do that if we want to debug.
#     """
#     steps = [init_a]  # history tracking
#     a = init_a

#     for _ in range(max_iter):
#         if a >= 0:
#             # gradient must be a function of a.

#             diff = learn_rate * r_squared_gradient(
#                 curve_trajectory, times_train, times_val, degree, a
#             )
#             if np.abs(diff) < tol:
#                 break
#             if a + diff < 0:
#                 break
#             a = a + diff
#             steps.append(a)  # history tracing

#     return a


def r_squared(curve_trajectory, times_train, times_val, degree, a):
    """Compute r squared."""
    b = 0.5
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    fit_variation = mse(
        curve_trajectory, elastic_metric, times_train, times_val, degree, a
    )
    total_variation = var(curve_trajectory, elastic_metric, times_val, a)

    return 1 - fit_variation / total_variation


def know_m_find_best_a(trajectory, degree, times_train, times_val, init_a, learn_rate):
    """Use a gradient search to find best a, for a given m.

    This function takes a trajectory and a degree and then uses gradient
    descent to find out which value of a minimizes the mean squared error
    (MSE) function.
    """
    max_iter = 100
    tol = 0.01
    return gradient_descent(
        init_a, learn_rate, max_iter, trajectory, times_train, times_val, degree, tol
    )


def find_best_am(curve_trajectory, init_a=0.2, n_train=10, n_val=10, learn_rate=0.1):
    """For a given geodesic, find the (m,a) pair that minimnizes rmse.

    Use a grid search on m and a gradient search on a to find the best pairs of (m,a).
    Then, choose the pair that minimizes root mean squared error.
    """
    # want to start with degree of 1 becuase that is a line, which is a geodesic
    ms = np.arange(1, 6)
    ms = np.array(ms)
    r2 = np.empty([len(ms)])
    best_as = np.empty([len(ms)])

    n_times = len(curve_trajectory)
    times = gs.arange(1, n_times + 1, 1)

    times_train = times[:n_train]  # noqa: E203
    times_val = times[n_train : (n_train + n_val)]  # noqa: E203
    times_test = times[(n_train + n_val) :]  # noqa: E203

    print(times_train)
    print(times_val)
    print(times_test)

    for i_m, degree in enumerate(ms):
        best_as[i_m] = know_m_find_best_a(
            curve_trajectory, degree, times_train, times_val, init_a, learn_rate
        )
        r2[i_m] = r_squared(
            curve_trajectory, times_train, times_val, degree, best_as[i_m]
        )
        print(
            "degree: "
            + str(degree)
            + "; best a: "
            + str(best_as[i_m])
            + ";r2: "
            + str(r2[i_m])
        )

    # r2 is the best when it is closest to +1.
    min_diff = 100
    i_best_r2 = 100
    for i_r2 in range(len(r2)):
        diff = r2[i_r2] - 1
        if abs(diff) < min_diff:
            min_diff = abs(r2[i_r2] - 1)
            i_best_r2 = i_r2

    best_am = gs.stack([best_as[i_best_r2], ms[i_best_r2]], axis=-1)

    print("best_a: " + str(best_am[0]) + " best_m: " + str(best_am[1]))
