"""Find the optimal a and m for a given trajectory.

a is an elastic metric parameter, and m is the degree of polynomial
regression fitting.

if given a trajectory, this will be able to
- find the best (a*,degree*) pair for fitting the trajectory.
- compare the r2 value of (a*,degree*) to the SRV geodesic
"""

import os

import geomstats.backend as gs
import numpy as np
from geomstats.geometry.discrete_curves import R2, ElasticMetric

os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def tau_matrix(times_train, m_degree):
    """Calculate tau matrix.

    tau = (X^T*X)^-1*X^T
    """
    X = np.empty([len(times_train), m_degree + 1])

    for i_time, time in enumerate(times_train):
        # Note: range(m_degree + 1) goes from 0 to m
        for i_degree in range(m_degree + 1):
            X[i_time][i_degree] = time**i_degree

    X_T = X.transpose()

    return np.linalg.inv(X_T @ X) @ X_T  # @ is matrix multiplication


def tau_sum(times_train, m_degree, i_val, j_train, times):
    """Calculate tau_ij.

    tau_sum is the sum of a bunch of tau's.
    tau_sum = sum_{l=0}^m tau_jl * t_i ** l

    tau_sum sums over tau matrices of increasing degrees,
    up to the highest degree. The highest degree is the degree
    of polynomial that this whole program is fitting

    TODO: more descriptive parameters
    variables:
    degree: polynomial degree
    l: the sum over degrees
    """
    # Note: Compute the tau matrix only once:
    tau_mat = tau_matrix(times_train, m_degree)

    tau_sum = 0
    # Note: range(m_degree + 1) goes from 0 to m
    for l_degree in range(m_degree + 1):
        tau_sum += tau_mat[l_degree, j_train] * times[i_val] ** l_degree

    return tau_sum


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


def dr_mse_da(i_val, trajectory, elastic_metric, times_train, degree):
    """Calculate the derivative of r_mse for a given i_val w.r.t. a.

    Utilizes f_transform in discrete_curves to calculate
    (sqrt of the norm of c prime) multiplied by (capital c
    to the power of a). remember that when we initialized
    elastic_metric, we set b= 0.5.

    Returns
    -------
    dr_da: the derivative of r_mse w.r.t. a.
    """
    n_times = len(trajectory)
    times = gs.arange(0, n_times, 1)

    fit_sum = 0
    for time_train in times_train:

        fit_sum += tau_sum(
            times_train, degree, i_val, time_train, times
        ) * derivative_q_curve(trajectory[time_train], elastic_metric)

    return derivative_q_curve(trajectory[i_val], elastic_metric) - fit_sum


def r_mse(i_val, trajectory, elastic_metric, times_train, degree):
    """Calculate r_mse for a given i_val (i.e. a given s parameter)."""
    n_times = len(trajectory)
    times = gs.arange(0, n_times, 1)

    fit_sum = 0
    for time_train in times_train:

        fit_sum += tau_sum(
            times_train, degree, i_val, time_train, times
        ) * elastic_metric.f_transform(trajectory[time_train])

    return elastic_metric.f_transform(trajectory[i_val]) - fit_sum


def d_mse(trajectory, elastic_metric, times_train, times_val, degree, a):
    """Compute the derivative of the MSE function w.r.t. a.

    TODO:
    put more descriptive caption later.
    explain why we calculate the dot product of r this way.

    we must multiply each element by (1/the number of sampling points -1)
    because we are
    performing a riemannian integral sum of the norm of r_i(s)'s.
    where i is the sampling point, and s is the parameter that
    parametrizes all curves and ranges from 0 to 1.
    the number of sampling points reduces by 1 because we are operating
    in q space.

    Parameters
    ----------
    r : is the distance between a q curve and the q curve expected from the
        regression fit
    dr_da : the derivative of the distance between a q curve and the q curve
        expected from the regression fit
    """
    n_sampling_points = len(trajectory[0][:, 0])

    d_mse_sum = 0

    for time_val in times_val:
        dr_da = dr_mse_da(time_val, trajectory, elastic_metric, times_train, degree)
        r = r_mse(time_val, trajectory, elastic_metric, times_train, degree)

        rows, cols = dr_da.shape

        for row in range(rows):
            for col in range(cols):

                d_mse_sum += dr_da[row][col] * r[row][col] / (n_sampling_points - 1)

    return 2 * d_mse_sum


def mse(trajectory, elastic_metric, times_train, times_val, degree, a):
    """Compute the mean squared error (MSE) with the given parameters.

    Computes the sum of the distances between each datapoint and its
    corresponding predicted datapoint.

    Parameters
    ----------
    r: is the distance between a q curve and the q curve expected from the
        regression fit
    dr_da: the derivative of the distance between a q curve and the q curve
        expected from the regression fit
    """
    n_sampling_points = len(trajectory[0][:, 0])

    mse_sum = 0

    for time_val in times_val:
        r = r_mse(
            i_val=time_val,
            trajectory=trajectory,
            elastic_metric=elastic_metric,
            times_train=times_train,
            degree=degree,
        )

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


def var(trajectory, elastic_metric, times_val, a):
    """Compute the total variation of the validation dataset.

    computes sum_i_train(norm(q_curve_i - mean_q_curve))

    Parameters
    ----------
    r : is the distance between a q curve and the mean q curve
    dr_da : the derivative of the distance between a q curve and the mean q curve.
    """
    n_sampling_points = len(trajectory[0][:, 0])

    q_mean = 0
    for time_val in times_val:
        q_mean += elastic_metric.f_transform(trajectory[time_val])

    q_mean = q_mean / len(times_val)

    var_sum = 0
    for time_val in times_val:
        r = r_var(trajectory[time_val], elastic_metric, q_mean)

        rows, cols = r.shape

        for row in range(rows):
            for col in range(cols):
                var_sum += r[row][col] * r[row][col] / (n_sampling_points - 1)

    return var_sum


def d_var(trajectory, elastic_metric, times_val, a):
    """Compute the total variation of the validation dataset w.r.t. a.

    computes derivative of sum_i_train(norm(q_curve_i - mean_q_curve))

    Parameters
    ----------
    r: is the distance between a q curve and the mean q curve
    dr_da: the derivative of the distance between a q curve and the mean q curve.
    """
    n_sampling_points = len(trajectory[0][:, 0])

    derivative_q_mean = 0
    for time_val in times_val:
        derivative_q_mean += derivative_q_curve(trajectory[time_val], elastic_metric)
    derivative_q_mean = derivative_q_mean / len(times_val)

    q_mean = 0
    for time_val in times_val:
        q_mean += elastic_metric.f_transform(trajectory[time_val])
    q_mean = q_mean / len(times_val)

    d_var_sum = 0
    for time_val in times_val:
        dr_da = dr_var_da(trajectory[time_val], elastic_metric, derivative_q_mean)
        r = r_var(trajectory[time_val], elastic_metric, q_mean)

        rows, cols = dr_da.shape

        for row in range(rows):
            for col in range(cols):

                d_var_sum += dr_da[row][col] * r[row][col] / (n_sampling_points - 1)

    return 2 * d_var_sum


def r_squared_gradient(trajectory, times_train, times_val, m, a):
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
        trajectory=trajectory,
        elastic_metric=elastic_metric,
        times_train=times_train,
        times_val=times_val,
        degree=m,
        a=a,
    )
    d_fit_variation = d_mse(trajectory, elastic_metric, times_train, times_val, m, a)
    total_variation = var(trajectory, elastic_metric, times_val, a)
    d_total_variation = d_var(trajectory, elastic_metric, times_val, a)

    gradient = (
        d_fit_variation * total_variation - fit_variation * d_total_variation
    ) / total_variation**2

    # Note: it would be better to have these prints outside of this function
    # because they are not directly related to the computation of the gradient
    r2_train = r_squared(trajectory, times_train, times_train, m, a)
    r2_val = r_squared(trajectory, times_train, times_val, m, a)
    mse_train = mse(trajectory, elastic_metric, times_train, times_train, m, a)
    mse_val = fit_variation

    print(
        f"a: {a:.3f}, r2_train: {r2_train:.3f}, r2_val: {r2_val:.3f}, "
        f"mse_train: {mse_train:.3f}, mse_val: {mse_val:.3f}, "
        f"total_variation: {total_variation:.3f}"
    )

    return gradient


def gradient_descent(
    a_init,
    a_lr,
    max_iter,
    min_iter,
    trajectory,
    times_train,
    times_val,
    times_test,
    degree,
    tol,
):
    """Calculate minimum x using gradient descent.

    structure inspiration source:
    https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21

    sample function also returns steps. we could do that if we want to debug.
    """
    a_steps = [a_init]  # history tracking
    elastic_metric = ElasticMetric(a_init, 0.5, ambient_manifold=R2)

    r2_train_steps = [r_squared(trajectory, times_train, times_train, degree, a_init)]
    r2_val_steps = [r_squared(trajectory, times_train, times_val, degree, a_init)]
    r2_test_steps = [r_squared(trajectory, times_train, times_test, degree, a_init)]
    mse_train_steps = [
        mse(
            trajectory=trajectory,
            elastic_metric=elastic_metric,
            times_train=times_train,
            times_val=times_train,
            degree=degree,
            a=a_init,
        )
    ]
    mse_val_steps = [
        mse(
            trajectory=trajectory,
            elastic_metric=elastic_metric,
            times_train=times_train,
            times_val=times_val,
            degree=degree,
            a=a_init,
        )
    ]
    mse_test_steps = [
        mse(
            trajectory=trajectory,
            elastic_metric=elastic_metric,
            times_train=times_train,
            times_val=times_test,
            degree=degree,
            a=a_init,
        )
    ]
    var_steps = [var(trajectory, elastic_metric, times_val, a_init)]

    a = a_init

    for i_iter in range(max_iter):
        if a >= 0:
            # gradient must be a function of a.

            diff = a_lr * r_squared_gradient(
                trajectory, times_train, times_val, degree, a
            )
            if np.abs(diff) < tol and i_iter > min_iter:
                break
            if a - diff < 0 and i_iter > min_iter:
                break
            a = a - diff

            # History tracking
            a_steps.append(a)
            elastic_metric = ElasticMetric(a, 0.5, ambient_manifold=R2)

            r2_train_steps.append(
                r_squared(trajectory, times_train, times_train, degree, a)
            )
            r2_val_steps.append(
                r_squared(trajectory, times_train, times_val, degree, a)
            )
            r2_test_steps.append(
                r_squared(trajectory, times_train, times_test, degree, a)
            )

            mse_train_steps.append(
                mse(
                    trajectory=trajectory,
                    elastic_metric=elastic_metric,
                    times_train=times_train,
                    times_val=times_train,
                    degree=degree,
                    a=a,
                )
            )
            mse_val_steps.append(
                mse(
                    trajectory=trajectory,
                    elastic_metric=elastic_metric,
                    times_train=times_train,
                    times_val=times_val,
                    degree=degree,
                    a=a,
                )
            )
            mse_test_steps.append(
                mse(
                    trajectory=trajectory,
                    elastic_metric=elastic_metric,
                    times_train=times_train,
                    times_val=times_test,
                    degree=degree,
                    a=a,
                )
            )
            var_steps.append(var(trajectory, elastic_metric, times_val, a))

    iteration_history = {}
    iteration_history["a"] = a_steps
    iteration_history["r2_train"] = r2_train_steps
    iteration_history["r2_val"] = r2_val_steps
    iteration_history["r2_test"] = r2_test_steps
    iteration_history["mse_train"] = mse_train_steps
    iteration_history["mse_val"] = mse_val_steps
    iteration_history["mse_test"] = mse_test_steps
    return a, iteration_history


def r_squared(trajectory, times_train, times_val, degree, a):
    """Compute r squared."""
    b = 0.5
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    fit_variation = mse(
        trajectory=trajectory,
        elastic_metric=elastic_metric,
        times_train=times_train,
        times_val=times_val,
        degree=degree,
        a=a,
    )
    total_variation = var(trajectory, elastic_metric, times_val, a)

    return 1 - fit_variation / total_variation


def know_m_find_best_a(
    trajectory, degree, times_train, times_val, times_test, a_init, a_lr, max_iter, tol
):
    """Use a gradient search to find best a, for a given m.

    This function takes a trajectory and a degree and then uses gradient
    descent to find out which value of a minimizes the mean squared error
    (MSE) function.
    """
    min_iter = 3
    print(
        f"DEGREE {degree}: Gradient ascent on R2 val wrt a "
        f"(a_lr={a_lr}, tol={tol}, max_iter={max_iter}):"
    )
    return gradient_descent(
        a_init=a_init,
        a_lr=a_lr,
        max_iter=max_iter,
        min_iter=min_iter,
        trajectory=trajectory,
        times_train=times_train,
        times_val=times_val,
        times_test=times_test,
        degree=degree,
        tol=tol,
    )


def find_best_am(
    trajectory,
    times_train=None,
    times_val=None,
    times_test=None,
    m_grid=None,
    a_init=0.2,
    a_lr=0.1,
    max_iter=20,
    tol=0.01,
):
    """For a given geodesic, find the (m,a) pair that maximizes R2.

    Use a grid search on m and a gradient search on a to find the best pairs of (m,a).
    Then, choose the pair that maximizes R2.
    """
    # want to start with degree of 1 because that is a line, which is a geodesic
    ms = gs.arange(1, 4) if m_grid is None else gs.array(m_grid)

    last_as_for_i_m = -gs.ones([len(ms)])
    r2_val_for_i_m = -gs.ones([len(ms)])
    r2_srv_val_for_i_m = -gs.ones([len(ms)])
    r2_test_for_i_m = -gs.ones([len(ms)])
    r2_srv_test_for_i_m = -gs.ones([len(ms)])
    iteration_histories_for_i_m = {}

    for i_m, degree in enumerate(ms):
        last_a, iteration_history = know_m_find_best_a(
            trajectory=trajectory,
            times_train=times_train,
            times_val=times_val,
            times_test=times_test,
            degree=degree,
            a_init=a_init,
            a_lr=a_lr,
            max_iter=max_iter,
            tol=tol,
        )

        r2_val = r_squared(
            trajectory=trajectory,
            times_train=times_train,
            times_val=times_val,
            degree=degree,
            a=last_a,
        )
        r2_srv_val = r_squared(
            trajectory=trajectory,
            times_train=times_train,
            times_val=times_val,
            degree=1,
            a=1,
        )
        r2_test = r_squared(
            trajectory=trajectory,
            times_train=times_train,
            times_val=times_test,
            degree=degree,
            a=last_a,
        )
        r2_srv_test = r_squared(
            trajectory=trajectory,
            times_train=times_train,
            times_val=times_test,
            degree=1,
            a=1,
        )

        print(
            f"--> DEGREE: {degree}; last_a: {last_a:.3f};"
            f" r2_val: {r2_val:.3f}; r2_srv_val: {r2_srv_val:.3f}"
            f" r2_test: {r2_test:.3f}; r2_srv_test: {r2_srv_test:.3f}"
        )

        last_as_for_i_m[i_m] = last_a
        iteration_histories_for_i_m[i_m] = iteration_history
        r2_val_for_i_m[i_m] = r2_val
        r2_srv_val_for_i_m[i_m] = r2_srv_val
        r2_test_for_i_m[i_m] = r2_test
        r2_srv_test_for_i_m[i_m] = r2_srv_test

    # r2 is the best when it is closest to +1.
    r2_val_diff_with_1 = gs.abs(r2_val_for_i_m - 1)
    best_i_m = gs.argmin(r2_val_diff_with_1)

    best_am = gs.stack([last_as_for_i_m[best_i_m], ms[best_i_m]], axis=-1)

    best_r2_val = r2_val_for_i_m[best_i_m]
    r2_srv_val_at_best_r2_val = r2_srv_val_for_i_m[best_i_m]
    r2_test_at_best_r2_val = r2_test_for_i_m[best_i_m]
    r2_srv_test_at_best_r2_val = r2_srv_test_for_i_m[best_i_m]

    print(
        "\n========> ACROSS DEGREES: Values corresponding to best R2:\n"
        f"best_a: {best_am[0]:.3f}\n"
        f"best_m: {best_am[1]:.3f}\n"
        f"best_r2_val: {best_r2_val:.3f}\n"
        f"r2_srv_val_at_best_r2_val: {r2_srv_val_at_best_r2_val:.3f}\n"
        f"r2_test_at_best_r2_val: {r2_test_at_best_r2_val:.3f}\n"
        f"r2_srv_test_at_best_r2_val: {r2_srv_test_at_best_r2_val:.3f}"
    )
    return (
        best_am[0],
        best_am[1],
        best_r2_val,
        r2_srv_val_at_best_r2_val,
        r2_test_at_best_r2_val,
        r2_srv_test_at_best_r2_val,
        iteration_histories_for_i_m,
    )
