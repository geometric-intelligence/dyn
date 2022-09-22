"""Visualization tools."""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


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


def plot_summary_wandb(
    iteration_histories_for_i_m,
    config,
    noiseless_curve_traj,
    curve_traj,
    noiseless_q_traj,
    q_traj,
    times_train,
    times_val,
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
    plot_name_to_h_val = {
        "a": a_true,
        "mse": 0,
        "r2": 1,
    }

    linestyle_dict = {
        "train": "-",
        "val": "--",
        "test": "-.",
    }

    # We can only see ~10 curves given the size of the plot
    factor = n_times // 10  # --> n_times // factor = 10
    fig = plt.figure(figsize=(20, 16), constrained_layout=True)

    gs = fig.add_gridspec(
        nrows=5, ncols=n_times // factor, height_ratios=[2, 1, 1, 1, 1]
    )
    ax_a = fig.add_subplot(gs[0, 0 : n_times // (factor * 3)])  # noqa: E203
    ax_mse = fig.add_subplot(
        gs[0, n_times // (factor * 3) : 2 * n_times // (factor * 3)]  # noqa: E203
    )
    ax_r2 = fig.add_subplot(
        gs[0, 2 * n_times // (factor * 3) : 3 * n_times // (factor * 3)]  # noqa: E203
    )
    axs = [ax_a, ax_mse, ax_r2]

    for i_plot, plot_name in enumerate(["a", "mse", "r2"]):
        if plot_name == "a":
            axs[i_plot].axhline(
                config.a_true, label=f"a_true = {config.a_true}", c="black"
            )
        for i_m, m in enumerate(config.m_grid):
            if plot_name == "a":
                iteration_history = iteration_histories_for_i_m[i_m][plot_name]
                iterations = np.arange(0, len(iteration_history))

                axs[i_plot].plot(
                    iterations, iteration_history, label=f"m = {m}", c=f"C{m}"
                )

            elif plot_name in ["mse", "r2"]:
                iteration_history = iteration_histories_for_i_m[i_m][
                    plot_name + "_train"
                ]
                iterations = np.arange(0, len(iteration_history))
                axs[i_plot].plot(
                    iterations,
                    iteration_history,
                    label=f"m = {m} (train)",
                    c=f"C{m}",
                    linestyle=linestyle_dict["train"],
                )
                iteration_history = iteration_histories_for_i_m[i_m][plot_name + "_val"]
                iterations = np.arange(0, len(iteration_history))
                axs[i_plot].plot(
                    iterations,
                    iteration_history,
                    label=f"m = {m} (val)",
                    c=f"C{m}",
                    linestyle=linestyle_dict["val"],
                )

                iteration_history = iteration_histories_for_i_m[i_m][
                    plot_name + "_test"
                ]
                iterations = np.arange(0, len(iteration_history))
                axs[i_plot].plot(
                    iterations,
                    iteration_history,
                    label=f"m = {m} (test)",
                    c=f"C{m}",
                    linestyle=linestyle_dict["test"],
                )

                axs[i_plot].axhline(plot_name_to_h_val[plot_name], c="black")

        axs[i_plot].set_xlabel("Iterations")
        axs[i_plot].set_title(plot_name)
        # axs[i_plot].set_ylim(plot_name_to_ylim[plot_name])

        axs[i_plot].legend()

    plt.suptitle(
        f"Ground truth: a_true = {a_true}, m_true = {m_true}  ---  "
        "Optimization a, m gives: "
        f"a = {best_a:.3f}, m = {best_m}, r2_val = {best_r2_val:.3f}  ---  "
        f"Evaluation: "
        f"r2_test = {r2_test_at_best:.3f}, "
        f"baseline_r2_srv_val = {baseline_r2_srv_val:.3f}",  # noqa: E501
        f"baseline_r2_srv_test = {baseline_r2_srv_test:.3f}",  # noqa: E501
        fontsize=18,
    )

    logging.info("3. Save plots of predicted curves.")
    for i_time in range(n_times // factor):
        noiseless_curve_ax = fig.add_subplot(gs[1, i_time])
        curve_ax = fig.add_subplot(gs[2, i_time])
        noiseless_q_ax = fig.add_subplot(gs[3, i_time])
        q_ax = fig.add_subplot(gs[4, i_time])
        if i_time == 0:
            noiseless_curve_ax.set_ylabel("Noiseless curve", fontsize=18)
            curve_ax.set_ylabel("Noisy curve", fontsize=18)
            noiseless_q_ax.set_ylabel("Noiseless q", fontsize=18)
            q_ax.set_ylabel("Noisy q", fontsize=18)
        elif factor * i_time >= n_train and factor * i_time < n_train + factor:
            noiseless_curve_ax.set_ylabel("Validation", fontsize=18)
            curve_ax.set_ylabel("Validation", fontsize=18)
            noiseless_q_ax.set_ylabel("Validation", fontsize=18)
            q_ax.set_ylabel("Validation", fontsize=18)
        elif (
            factor * i_time >= n_train + n_val
            and factor * i_time < n_train + n_val + factor
        ):
            noiseless_curve_ax.set_ylabel("Test", fontsize=18)
            curve_ax.set_ylabel("Test", fontsize=18)
            noiseless_q_ax.set_ylabel("Test", fontsize=18)
            q_ax.set_ylabel("Test", fontsize=18)
        noiseless_curve_ax.plot(
            noiseless_curve_traj[factor * i_time][:, 0],
            noiseless_curve_traj[factor * i_time][:, 1],
            marker="o",
            c="C0",
        )
        curve_ax.plot(
            curve_traj[factor * i_time][:, 0],
            curve_traj[factor * i_time][:, 1],
            marker="o",
            c="C1",
        )
        noiseless_q_ax.plot(
            noiseless_q_traj[factor * i_time][:, 0],
            noiseless_q_traj[factor * i_time][:, 1],
            marker="o",
            c="C0",
        )
        q_ax.plot(
            q_traj[factor * i_time][:, 0],
            q_traj[factor * i_time][:, 1],
            marker="o",
            c="C1",
        )
    return fig
