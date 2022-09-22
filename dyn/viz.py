"""Visualization tools."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geomstats.geometry.discrete_curves import R2, ElasticMetric

import dyn.dyn.features.math_am as math_am

LINESTYLE_DICT = {
    "train": "-",
    "val": "--",
    "test": "-.",
}

MARKER_DICT = {
    "curve": "o",
    "q": "x",
}


def init_matplotlib():
    """Configure style for matplotlib tutorial."""
    fontsize = 18
    matplotlib.rc("font", size=fontsize)
    matplotlib.rc("text")
    matplotlib.rc("legend", fontsize=fontsize)
    matplotlib.rc("axes", titlesize=21, labelsize=14)
    matplotlib.rc(
        "font",
        family="sans-serif",
        monospace=["Arial"],
    )


def plot_a_steps(axs, i_plot, i_m, m, iteration_histories_per_i_m):
    """Plot a steps."""
    iteration_history = iteration_histories_per_i_m[i_m]["a"]
    iterations = np.arange(0, len(iteration_history))

    axs[i_plot].plot(iterations, iteration_history, label=f"m = {m}", c=f"C{m}")
    return axs


def plot_mse_or_r2_steps(axs, i_plot, i_m, m, iteration_histories_per_i_m, plot_name):
    """Plot mse or r2 steps."""
    iteration_history = iteration_histories_per_i_m[i_m][plot_name + "_train"]
    iterations = np.arange(0, len(iteration_history))
    axs[i_plot].plot(
        iterations,
        iteration_history,
        label=f"m = {m} (train)",
        c=f"C{m}",
        linestyle=LINESTYLE_DICT["train"],
    )
    iteration_history = iteration_histories_per_i_m[i_m][plot_name + "_val"]
    iterations = np.arange(0, len(iteration_history))
    axs[i_plot].plot(
        iterations,
        iteration_history,
        label=f"m = {m} (val)",
        c=f"C{m}",
        linestyle=LINESTYLE_DICT["val"],
    )

    iteration_history = iteration_histories_per_i_m[i_m][plot_name + "_test"]
    iterations = np.arange(0, len(iteration_history))
    axs[i_plot].plot(
        iterations,
        iteration_history,
        label=f"m = {m} (test)",
        c=f"C{m}",
        linestyle=LINESTYLE_DICT["test"],
    )
    return axs


def plot_summary_wandb(
    iteration_histories_per_i_m,
    config,
    noiseless_curve_traj,
    curve_traj,
    noiseless_q_traj,
    q_traj,
    times_train,
    times_val,
    times_test,
    best_a,
    best_m,
    best_r2_val,
    r2_test_at_best,
    baseline_r2_srv_val,
    baseline_r2_srv_test,
):
    """Save the master figure in wandb."""
    a_true = config.a_true
    m_true = config.m_true
    n_times = config.n_times
    n_train = len(times_train)
    n_val = len(times_val)
    # plot_name_to_ylim = {
    #     "a": (0.75, 1.),
    #     "mse": (0, 1),
    #     "r2": (-3, 1.1),
    # }

    # Make predictions
    elastic_metric = ElasticMetric(a=best_a, b=0.5, ambient_manifold=R2)
    coeffs = math_am.coeffs(times_train, curve_traj, elastic_metric, m=best_m)
    times = np.concatenate([times_train, times_val, times_test])
    assert times.shape[0] == n_times, times.shape
    q_pred_traj = math_am.predict_q(times, coeffs)

    starting_point_array = np.zeros((len(times), 2))
    curve_pred_traj = elastic_metric.f_transform_inverse(
        q_pred_traj, starting_point_array
    )

    # Make baseline predictions with geodesic (m=1) of srv metric (a=1)
    srv_metric = ElasticMetric(a=1, b=0.5, ambient_manifold=R2)
    coeffs = math_am.coeffs(times_train, curve_traj, srv_metric, m=1)
    times = np.concatenate([times_train, times_val, times_test])
    assert times.shape[0] == n_times, times.shape
    srv_q_pred_traj = math_am.predict_q(times, coeffs)

    starting_point_array = np.zeros((len(times), 2))
    srv_curve_pred_traj = srv_metric.f_transform_inverse(
        srv_q_pred_traj, starting_point_array
    )

    # We can only see ~10 curves given the size of the plot
    factor = n_times // 10  # --> n_times // factor = 10
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)

    # Figure has 9 rows:
    # - 1 row for a, mse, and r2
    # - 4 rows for the data: in curve space and in q space, noiseless and noisy
    # - 2 rows for the prediction: in curve space and in q space
    # - 2 rows for the baseline prediction: in curve space and in q space

    gs = fig.add_gridspec(
        nrows=9, ncols=n_times // factor, height_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1]
    )

    # Plot first row: a, mse, and r2
    ax_a = fig.add_subplot(gs[0, 0 : n_times // (factor * 3)])  # noqa: E203
    ax_mse = fig.add_subplot(
        gs[0, n_times // (factor * 3) : 2 * n_times // (factor * 3)]  # noqa: E203
    )
    ax_r2 = fig.add_subplot(
        gs[0, 2 * n_times // (factor * 3) : 3 * n_times // (factor * 3)]  # noqa: E203
    )
    axs = [ax_a, ax_mse, ax_r2]

    for i_plot, plot_name in enumerate(["a", "mse", "r2"]):
        for i_m, m in enumerate(config.m_grid):
            if plot_name == "a":
                axs = plot_a_steps(axs, i_plot, i_m, m, iteration_histories_per_i_m)

            elif plot_name in ["mse", "r2"]:
                axs = plot_mse_or_r2_steps(
                    axs, i_plot, i_m, m, iteration_histories_per_i_m, plot_name
                )

        plot_name_to_h_val = {"a": config.a_true, "mse": 0, "r2": 1}
        axs[i_plot].axhline(plot_name_to_h_val[plot_name], c="black")
        axs[i_plot].set_xlabel("Iterations")
        axs[i_plot].set_title(plot_name)
        # axs[i_plot].set_ylim(plot_name_to_ylim[plot_name])

        axs[i_plot].legend()

    plt.suptitle(
        f"Ground truth: a_true = {a_true}, m_true = {m_true}  ---  "
        f"Optimizing ({config.a_optimization}) a, m gives: "
        f"a = {best_a:.3f}, m = {best_m}, r2_val = {best_r2_val:.3f}  ---  "
        f"Evaluation: "
        f"r2_test = {r2_test_at_best:.3f}, "
        f"baseline_r2_srv_val = {baseline_r2_srv_val:.3f}, "  # noqa: E501
        f"baseline_r2_srv_test = {baseline_r2_srv_test:.3f}",  # noqa: E501
        fontsize=14,
    )

    # Plot 4 rows of data + 2 rows of prediction
    for i_time in range(n_times // factor):
        noiseless_curve_ax = fig.add_subplot(gs[1, i_time])
        curve_ax = fig.add_subplot(gs[2, i_time])
        noiseless_q_ax = fig.add_subplot(gs[3, i_time])
        q_ax = fig.add_subplot(gs[4, i_time])
        curve_pred_ax = fig.add_subplot(gs[5, i_time])
        q_pred_ax = fig.add_subplot(gs[6, i_time])
        srv_curve_pred_ax = fig.add_subplot(gs[7, i_time])
        srv_q_pred_ax = fig.add_subplot(gs[8, i_time])
        if i_time == 0:
            noiseless_curve_ax.set_ylabel("Noiseless curve", fontsize=18)
            curve_ax.set_ylabel("Noisy curve", fontsize=18)
            noiseless_q_ax.set_ylabel("Noiseless q", fontsize=18)
            q_ax.set_ylabel("Noisy q", fontsize=18)
            curve_pred_ax.set_ylabel("Pred curve", fontsize=18)
            q_pred_ax.set_ylabel("Pred q", fontsize=18)
            srv_curve_pred_ax.set_ylabel("SRV pred curve", fontsize=18)
            srv_q_pred_ax.set_ylabel("SRV pred q", fontsize=18)
        elif factor * i_time >= n_train and factor * i_time < n_train + factor:
            noiseless_curve_ax.set_ylabel("Validation", fontsize=18)
            curve_ax.set_ylabel("Validation", fontsize=18)
            noiseless_q_ax.set_ylabel("Validation", fontsize=18)
            q_ax.set_ylabel("Validation", fontsize=18)
            curve_pred_ax.set_ylabel("Validation", fontsize=18)
            q_pred_ax.set_ylabel("Validation", fontsize=18)
            srv_curve_pred_ax.set_ylabel("Validation", fontsize=18)
            srv_q_pred_ax.set_ylabel("Validation", fontsize=18)
        elif (
            factor * i_time >= n_train + n_val
            and factor * i_time < n_train + n_val + factor
        ):
            noiseless_curve_ax.set_ylabel("Test", fontsize=18)
            curve_ax.set_ylabel("Test", fontsize=18)
            noiseless_q_ax.set_ylabel("Test", fontsize=18)
            q_ax.set_ylabel("Test", fontsize=18)
            curve_pred_ax.set_ylabel("Test", fontsize=18)
            q_pred_ax.set_ylabel("Test", fontsize=18)
            srv_curve_pred_ax.set_ylabel("Test", fontsize=18)
            srv_q_pred_ax.set_ylabel("Test", fontsize=18)

        # Plot 4 rows of data
        noiseless_curve_ax.plot(
            noiseless_curve_traj[factor * i_time][:, 0],
            noiseless_curve_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["curve"],
            c="C0",
        )
        curve_ax.plot(
            curve_traj[factor * i_time][:, 0],
            curve_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["curve"],
            c="C1",
        )
        noiseless_q_ax.plot(
            noiseless_q_traj[factor * i_time][:, 0],
            noiseless_q_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["q"],
            c="C0",
        )
        q_ax.plot(
            q_traj[factor * i_time][:, 0],
            q_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["q"],
            c="C1",
        )

        # Plot 2 rows of predictions
        curve_pred_ax.plot(
            curve_pred_traj[factor * i_time][:, 0],
            curve_pred_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["curve"],
            c="C2",
        )
        q_pred_ax.plot(
            q_pred_traj[factor * i_time][:, 0],
            q_pred_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["q"],
            c="C2",
        )

        # Plot 2 rows of SRV predictions
        srv_curve_pred_ax.plot(
            srv_curve_pred_traj[factor * i_time][:, 0],
            srv_curve_pred_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["curve"],
            c="C3",
        )
        srv_q_pred_ax.plot(
            srv_q_pred_traj[factor * i_time][:, 0],
            srv_q_pred_traj[factor * i_time][:, 1],
            marker=MARKER_DICT["q"],
            c="C3",
        )

    return fig
