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
import wandb
from geomstats.geometry.discrete_curves import R2, ElasticMetric

import dyn.dyn.features.math_am as math_am

os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def log_iteration(iter, max_iter, trajectory, times_train, times_val, times_test, m, a):
    """Log the iteration."""
    elastic_metric = ElasticMetric(a, b=0.5, ambient_manifold=R2)

    r2_train = math_am.r_squared(trajectory, times_train, times_train, m, a)
    r2_val = math_am.r_squared(trajectory, times_train, times_val, m, a)
    r2_test = math_am.r_squared(trajectory, times_train, times_test, m, a)
    mse_train = math_am.mse(trajectory, elastic_metric, times_train, times_train, m, a)
    mse_val = math_am.mse(trajectory, elastic_metric, times_train, times_val, m, a)
    mse_test = math_am.mse(trajectory, elastic_metric, times_train, times_test, m, a)

    print(
        f"i_iter: {iter}/{max_iter}, a: {a:.3f}\n"
        f"   r2_train: {r2_train:.3f}, r2_val*: {r2_val:.3f}, r2_test: {r2_test:.3f}\n"  # noqa: E501
        f"   mse_train: {mse_train:.3f}, mse_val: {mse_val:.3f}, mse_test: {mse_test:.3f}"  # noqa: E501
    )

    wandb.log(
        {
            "a": a,
            "train": {
                "r2_train": r2_train,
                "mse_train": mse_train,
            },
            "val": {
                "r2_val": r2_val,
                "mse_val": mse_val,
            },
            "test": {
                "r2_test": r2_test,
                "mse_test": mse_test,
            },
        },
        step=iter,
    )
    return r2_train, r2_val, r2_test, mse_train, mse_val, mse_test


def gradient_update_rule(trajectory, times_train, times_val, degree, a):
    """Calculate the gradient descent step."""
    update_rule = math_am.r_squared_gradient(
        trajectory, times_train, times_val, degree, a
    )
    return update_rule


def adam_update_rule(
    iter, moment_m, moment_v, trajectory, times_train, times_val, degree, a
):
    """Calculate the adam descent step."""
    # Adam parameters: default values work well.
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    gradient = math_am.r_squared_gradient(trajectory, times_train, times_val, degree, a)
    moment_m = beta1 * moment_m + (1 - beta1) * gradient
    moment_v = beta2 * moment_v + (1 - beta2) * (gradient**2)
    moment_m_hat = moment_m / (1 - beta1**iter)
    moment_v_hat = moment_v / (1 - beta2**iter)
    update_rule = (moment_m_hat / (np.sqrt(moment_v_hat) + epsilon),)
    return update_rule, moment_m, moment_v


def know_m_find_best_a(
    trajectory,
    m,
    times_train,
    times_val,
    times_test,
    a_init,
    a_lr,
    max_iter,
    tol,
    a_optimization="sgd",
):
    """Use a gradient search to find best a, for a given m.

    This function takes a trajectory and a degree and then uses gradient
    descent to find out which value of a minimizes the mean squared error
    (MSE) function.
    """
    min_iter = 3
    print(
        f"DEGREE {m}: Gradient ascent on R2 val wrt a "
        f"(a_lr={a_lr}, tol={tol}, max_iter={max_iter}):"
    )
    elastic_metric = ElasticMetric(a_init, 0.5, ambient_manifold=R2)

    a_steps = [a_init]
    r2_train_steps = [
        math_am.r_squared(trajectory, times_train, times_train, m, a_init)
    ]
    r2_val_steps = [math_am.r_squared(trajectory, times_train, times_val, m, a_init)]
    r2_test_steps = [math_am.r_squared(trajectory, times_train, times_test, m, a_init)]
    mse_train_steps = [
        math_am.mse(trajectory, elastic_metric, times_train, times_train, m, a_init)
    ]
    mse_val_steps = [
        math_am.mse(trajectory, elastic_metric, times_train, times_val, m, a_init)
    ]
    mse_test_steps = [
        math_am.mse(trajectory, elastic_metric, times_train, times_test, m, a_init)
    ]

    # initiate adam parameters
    moment_m = 0
    moment_v = 0

    a = a_init

    for iter in range(max_iter):
        if a >= 0:
            if a_optimization == "sgd":
                update_rule = gradient_update_rule(
                    trajectory, times_train, times_val, m, a
                )
                diff = a_lr * update_rule
            elif a_optimization == "adam":
                update_rule, moment_m, moment_v = adam_update_rule(
                    iter, moment_m, moment_v, trajectory, times_train, times_val, m, a
                )
                diff = a_lr * update_rule
            if np.abs(diff) < tol and iter > min_iter:
                break

            # Note: minus sign -> gradient descent
            a = a - diff

            # History tracking: track every 5 iterations for efficiency
            if iter % 5 == 0 or iter == max_iter - 1:
                r2_train, r2_val, r2_test, mse_train, mse_val, mse_test = log_iteration(
                    iter, max_iter, trajectory, times_train, times_val, times_test, m, a
                )
                a_steps.append(a)
                r2_train_steps.append(r2_train)
                r2_val_steps.append(r2_val)
                r2_test_steps.append(r2_test)
                mse_train_steps.append(mse_train)
                mse_val_steps.append(mse_val)
                mse_test_steps.append(mse_test)

    iteration_history = {}
    iteration_history["a"] = a_steps
    iteration_history["r2_train"] = r2_train_steps
    iteration_history["r2_val"] = r2_val_steps
    iteration_history["r2_test"] = r2_test_steps
    iteration_history["mse_train"] = mse_train_steps
    iteration_history["mse_val"] = mse_val_steps
    iteration_history["mse_test"] = mse_test_steps
    return a, iteration_history


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

    iteration_histories_per_i_m = {}

    for i_m, m in enumerate(ms):
        last_a, iteration_history = know_m_find_best_a(
            trajectory=trajectory,
            times_train=times_train,
            times_val=times_val,
            times_test=times_test,
            m=m,
            a_init=a_init,
            a_lr=a_lr,
            max_iter=max_iter,
            tol=tol,
        )

        r2_val = iteration_history["r2_val"][-1]
        r2_test = iteration_history["r2_test"][-1]

        print(
            f"--> DEGREE: {m}; last_a: {last_a:.3f};"
            f" r2_val: {r2_val:.3f}; r2_test: {r2_test:.3f};"
        )

        iteration_histories_per_i_m[i_m] = iteration_history

    # r2 is the best when it is closest to +1.
    r2_val_per_i_m = np.array(
        [
            iteration_history["r2_val"][-1]
            for iteration_history in iteration_histories_per_i_m.values()
        ]
    )
    r2_val_diff_with_1 = np.abs(r2_val_per_i_m - 1)
    best_i_m = np.argmin(r2_val_diff_with_1)

    best_a = iteration_histories_per_i_m[best_i_m]["a"][-1]
    best_m = ms[best_i_m]

    best_r2_val = iteration_histories_per_i_m[best_i_m]["r2_val"][-1]
    r2_test_at_best = iteration_histories_per_i_m[best_i_m]["r2_test"][-1]

    # Comparison with baseline: geodesic (m=1) and default srv (a=1)
    baseline_r2_srv_val = math_am.r_squared(
        trajectory, times_train, times_val, degree=1, a=1
    )
    baseline_r2_srv_test = math_am.r_squared(
        trajectory, times_train, times_test, degree=1, a=1
    )

    print(
        "\n========> ACROSS DEGREES: Values corresponding to best R2:\n"
        f"best_a: {best_a:.3f}; best_m: {best_m};\n"
        f"best_r2_val: {best_r2_val:.3f};\n"
        f"r2_test_at_best: {r2_test_at_best:.3f}\n"
        f"compare with: r2_srv_val: {baseline_r2_srv_val:.3f}\n"
        f"compare with: r2_srv_test: {baseline_r2_srv_test:.3f}\n"
    )
    return (
        best_a,
        best_m,
        best_r2_val,
        r2_test_at_best,
        baseline_r2_srv_val,
        baseline_r2_srv_test,
        iteration_histories_per_i_m,
    )
