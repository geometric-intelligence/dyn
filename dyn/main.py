"""Main script."""

import itertools
import logging
import os
import tempfile

import default_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

import dyn.dyn.datasets.experimental as experimental
import dyn.dyn.datasets.synthetic as synthetic
import dyn.dyn.features.optimize_am as optimize_am

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

np.random.seed(2022)

# i feel like we should load the dataset once at the
# beginning of the script, and then
# - change number of sampling points later,
# - can directly choose number of time points in synthetic
# - can give it noise maybe in synthetic.


def run_tests():
    """Run wandb with different input parameters and tests."""
    # This is the equivalent of several for-loops:
    for (
        dataset_name,
        a_true,
        m_true,
        n_sampling_points,
        noise_var,
        n_times,
        a_init_diff,
    ) in itertools.product(
        default_config.dataset_name,
        default_config.a_true,
        default_config.m_true,
        default_config.n_sampling_points,
        default_config.noise_var,
        default_config.n_times,
        default_config.a_init_diff,
    ):
        logging.info(
            f"Running tests for {dataset_name} with a={a_true}, m={m_true}:\n"
            f"- n_sampling_points={n_sampling_points}\n"
            f"- noise_var={noise_var}\n"
            f"- n_times={n_times}\n"
        )

        if dataset_name == "cells":
            cells = [default_config.start_cell, default_config.end_cell]
            # Note: using n_cells = -1 avoid the random selection of cells
            _, cell_shapes, labels_a, labels_b = experimental.preprocess(
                cells=cells,
                labels_a=default_config.lines,
                labels_b=default_config.treatments,
                n_cells=-1,
                n_sampling_points=n_sampling_points,
                quotient=default_config.quotient,
            )
            start_cell = cell_shapes[0]
            end_cell = cell_shapes[1]

        elif dataset_name == "circles":
            circle_trajectory = synthetic.geodesics_circle_to_ellipse(
                n_geodesics=1, n_times=2, n_points=n_sampling_points
            )
            circle_trajectory = circle_trajectory[0]
            start_cell = circle_trajectory[0]
            end_cell = circle_trajectory[1]
        else:
            raise ValueError(f"Unknown dataset name {dataset_name}")

        a_init = a_true + a_init_diff

        run_wandb(
            dataset_name=dataset_name,
            a_true=a_true,
            m_true=m_true,
            n_times=n_times,
            n_sampling_points=n_sampling_points,
            noise_var=noise_var,
            a_init=a_init,
            start_cell=start_cell,
            end_cell=end_cell,
        )


def run_wandb(
    dataset_name,
    a_true,
    m_true,
    n_times,
    n_sampling_points,
    noise_var,
    a_init,
    start_cell,
    end_cell,
):
    """Run wandb script for the following parameters."""
    run_name = (
        f"{dataset_name}_at{a_true}_ai{a_init}_mt{m_true}_"
        + f"nt{n_times}_nsp{n_sampling_points}_nv{noise_var}_{default_config.now}"
    )

    wandb.init(
        project="metric_learning",
        dir=tempfile.gettempdir(),
        config={
            "run_name": run_name,
            "dataset_name": dataset_name,
            "a_true": a_true,
            "m_true": m_true,
            "noise_var": noise_var,
            "n_sampling_points": n_sampling_points,
            "n_times": n_times,
            "a_init": a_init,
            "m_grid": default_config.m_grid,
            "a_optimization": default_config.a_optimization,
            "a_lr": default_config.a_lr,
        },
    )

    config = wandb.config
    wandb.run.name = config.run_name

    logging.info(
        f"Load dataset {dataset_name} with " f"a_true = {a_true} and m_true = {m_true}"
    )
    b = 0.5
    trajectory_data = synthetic.geodesic_between_curves(
        start_cell, end_cell, a_true, b, n_times, noise_var
    )
    print(f"The shape of the trajectory is: {trajectory_data.shape}")

    logging.info("Find best a and m corresponding to the trajectory.")
    (
        best_a,
        best_m,
        best_r2_val,
        r2_srv_val_at_best_r2_val,
        r2_test_at_best_r2_val,
        r2_srv_test_at_best_r2_val,
        iteration_histories_for_i_m,
    ) = optimize_am.find_best_am(
        trajectory_data, a_init=config.a_init, m_grid=config.m_grid, a_lr=config.a_lr
    )

    logging.info("--->>> Save results in wandb and local saved_figs directory.")

    logging.info("1. Save the config locally.")
    config_df = pd.DataFrame.from_dict(dict(config))
    config_df.to_json(f"saved_figs/optimize_am/{config.run_name}_config.json")

    logging.info("2. Save iteration histories during gradient descent.")

    for i_m, m in enumerate(config.m_grid):
        a_steps = iteration_histories_for_i_m[i_m]["a"]
        mse_train_steps = iteration_histories_for_i_m[i_m]["mse_train"]
        mse_val_steps = iteration_histories_for_i_m[i_m]["mse_val"]

        r2_train_steps = iteration_histories_for_i_m[i_m]["r2_train"]
        r2_val_steps = iteration_histories_for_i_m[i_m]["r2_val"]

        iteration_history_df = pd.DataFrame(
            columns=["a", "mse_train", "mse_val", "r2_train", "r2_val"],
            data=[
                [
                    float(a),
                    float(mse_train),
                    float(mse_val),
                    float(r_train),
                    float(r_val),
                ]
                for a, mse_train, mse_val, r_train, r_val in zip(
                    a_steps,
                    mse_train_steps,
                    mse_val_steps,
                    r2_train_steps,
                    r2_val_steps,
                )
            ],
        )

        table_key = f"iteration_history_m_{m}"
        iteration_history_df.to_json(
            f"saved_figs/optimize_am/{config.run_name}_iteration_history.json"
        )
        wandb.log({table_key: wandb.Table(dataframe=iteration_history_df)})

    plot_name_to_ylim = {
        "a": (-0.1, 5),
        "mse": (-0.1, 5),
        "r2": (-1.1, 1.1),
    }
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

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
                    iterations, iteration_history, label=f"m = {m}", c=f"C{m-1}"
                )

            elif plot_name in ["mse", "r2"]:
                hval = 0 if plot_name == "mse" else 1
                iteration_history = iteration_histories_for_i_m[i_m][
                    plot_name + "_train"
                ]
                iterations = np.arange(0, len(iteration_history))
                axs[i_plot].plot(
                    iterations,
                    iteration_history,
                    label=f"m = {m} (train)",
                    c=f"C{m-1}",
                    linestyle="-",
                )
                iteration_history = iteration_histories_for_i_m[i_m][plot_name + "_val"]
                iterations = np.arange(0, len(iteration_history))
                axs[i_plot].plot(
                    iterations,
                    iteration_history,
                    label=f"m = {m} (val)",
                    c=f"C{m-1}",
                    linestyle="--",
                )

                axs[i_plot].axhline(hval, c="black")
                axs[i_plot].set_ylim(
                    min(iteration_histories_for_i_m[0][plot_name + "_val"]),
                    max(iteration_histories_for_i_m[0][plot_name + "_val"]),
                )

        axs[i_plot].set_xlabel("Iterations")
        axs[i_plot].set_title(plot_name)
        axs[i_plot].set_ylim(plot_name_to_ylim[plot_name])

        axs[i_plot].legend()

    fig.suptitle(
        "Optimization a, m gives: "
        f"a = {best_a:.3f}, m = {best_m}, r2_val = {best_r2_val:.3f}\t"
        f"Evaluation:"
        f"r2_test = {r2_test_at_best_r2_val:.3f}, r2_srv_test = {r2_srv_test_at_best_r2_val}"  # noqa: E501
    )
    fig.savefig(f"saved_figs/optimize_am/{config.run_name}_iteration_history.png")
    wandb.log({"optimization_fig": wandb.Image(fig)})

    logging.info("3. Save plots of predicted curves.")
    # TODO.

    wandb.finish()


run_tests()

# logging.info(f"Starting run {default_config.run_name}")
# wandb.init(
#     project="metric_learning",
#     dir=tempfile.gettempdir(),
#     config={
#         "run_name": default_config.run_name,
#         "dataset_name": default_config.dataset_name,
#         "a_true": default_config.a_true,
#         "m_true": default_config.m_true,
#         "noise_var": default_config.noise_var,
#         "n_sampling_points": default_config.n_sampling_points,
#         "n_times": default_config.n_times,
# #         "a_initialization": default_config.a_initialization,
#         "m_grid": default_config.m_grid,
#         "a_optimization": default_config.a_optimization,
#         "a_lr": default_config.a_lr,
#         "a_init": default_config.a_init,
#         "start_cell":default_config.start_cell,
#         "end_cell":default_config.end_cell
#     },
# )

# config = wandb.config

# wandb.run.name = config.run_name

# logging.info(
#     f"Load dataset {config.dataset_name} with "
#     f"a_true = {config.a_true} and m_true = {config.m_true}"
# )
# trajectory_data = None
# if config.dataset_name == "synthetic_geodesic_between_curves":
#     b = 0.5
#     print('start_cell: '+str(config.start_cell))
#     trajectory_data = synthetic.geodesic_between_curves(
#         config.start_cell, config.end_cell, config.a_true, b, config.m_true, config.n_times, config.noise_var # noqa: E501
#     )
# if trajectory_data is None:
#     raise NotImplementedError()

# # one_trajectory = dataset_of_trajectories[0]
# # print(f"The shape of one_trajectory is: {one_trajectory.shape}")

# # if config.a_initialization == "close_to_ground_truth":
# #     a_init = config.a_true - 0.2
# # elif config.a_initialization == "random":
# #     a_init = 0.5
# # else:
# #     raise NotImplementedError()

# logging.info("Find best a and m corresponding to the trajectory.")
# best_a, best_m, best_r2, r2, r2_srv, r2_test, r2_srv_test, iteration_histories = optimize_am.find_best_am( # noqa: E501
#     trajectory_data, a_init=config.a_init, m_grid=config.m_grid, a_lr=config.a_lr
# )

# logging.info("--->>> Save results in wandb and local saved_figs directory.")

# logging.info("1. Save the config locally.")
# config_df = pd.DataFrame.from_dict(dict(config))
# config_df.to_json(f"saved_figs/optimize_am/{config.run_name}_config.json")

# logging.info("2. Save best values for a, m and r2.")
# best_amr2_df = pd.DataFrame(
#     columns=["best_a", "best_m", "best_r2"], data=[[best_a, best_m, best_r2]]
# )

# r2s_from_m_df = pd.DataFrame(
#     columns=[f"m = {m}" for m in list(config.m_grid)], data=[list(r2), list(r2_srv)]
# )

# best_amr2_df.to_json(f"saved_figs/optimize_am/{config.run_name}_best_amr2.json")
# wandb.log({"best_amr2": wandb.Table(dataframe=best_amr2_df)})
# r2s_from_m_df.to_json(f"saved_figs/optimize_am/{config.run_name}_r2s_from_m_df.json")
# wandb.log({"r2s_from_m": wandb.Table(dataframe=r2s_from_m_df)})

# logging.info("3. Save iteration histories during gradient descent.")

# for i_m, m in enumerate(config.m_grid):
#     a_steps = iteration_histories[i_m]["a"]
#     mse_train_steps = iteration_histories[i_m]["mse_train"]
#     mse_val_steps = iteration_histories[i_m]["mse_val"]

#     r2_train_steps = iteration_histories[i_m]["r2_train"]
#     r2_val_steps = iteration_histories[i_m]["r2_val"]

#     iteration_history_df = pd.DataFrame(
#         columns=["a", "mse_train", "mse_val", "r2_train", "r2_val"],
#         data=[
#             [float(a), float(mse_train), float(mse_val), float(r_train), float(r_val)]
#             for a, mse_train, mse_val, r_train, r_val in zip(
#                 a_steps, mse_train_steps, mse_val_steps, r2_train_steps, r2_val_steps
#             )
#         ],
#     )

#     table_key = f"iteration_history_m_{m}"
#     iteration_history_df.to_json(
#         f"saved_figs/optimize_am/{config.run_name}_iteration_history.json"
#     )
#     wandb.log({table_key: wandb.Table(dataframe=iteration_history_df)})

# fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# for i_plot, plot_name in enumerate(["a", "mse", "r2"]):
#     if plot_name == "a":
#         axs[i_plot].axhline(config.a_true, label=f"a_true = {config.a_true}", c="black") # noqa: E501
#     for i_m, m in enumerate(config.m_grid):
#         if plot_name == "a":
#             iteration_history = iteration_histories[i_m][plot_name]
#             iterations = np.arange(0, len(iteration_history))
#             axs[i_plot].plot(
#                 iterations, iteration_history, label=f"m = {m}", c=f"C{m-1}"
#             )

#         elif plot_name in ["mse", "r2"]:
#             hval = 0 if plot_name == "mse" else 1
#             iteration_history = iteration_histories[i_m][plot_name + "_train"]
#             iterations = np.arange(0, len(iteration_history))
#             axs[i_plot].plot(
#                 iterations,
#                 iteration_history,
#                 label=f"m = {m} (train)",
#                 c=f"C{m-1}",
#                 linestyle="-",
#             )
#             iteration_history = iteration_histories[i_m][plot_name + "_val"]
#             iterations = np.arange(0, len(iteration_history))
#             axs[i_plot].plot(
#                 iterations,
#                 iteration_history,
#                 label=f"m = {m} (val)",
#                 c=f"C{m-1}",
#                 linestyle="--",
#             )

#             axs[i_plot].axhline(hval, c="black")
#     axs[i_plot].set_xlabel("Iterations")
#     axs[i_plot].set_title(plot_name)
#     axs[i_plot].legend()

# fig.suptitle(f"Optimization a, m gives: a = {best_a:.3f}, m = {best_m}, r2 = {best_r2}") # noqa: E501
# fig.savefig(f"saved_figs/optimize_am/{config.run_name}_iteration_history.png")
# wandb.log({"optimization_fig": wandb.Image(fig)})

# logging.info("4. Save plots of predicted curves.")
# # TODO.

# wandb.finish()


# logging.info(f"Starting run {default_config.run_name}")
# wandb.init(
#     project="metric_learning",
#     dir=tempfile.gettempdir(),
#     config={
#         "run_name": default_config.run_name,
#         "dataset_name": default_config.dataset_name,
#         "a_true": default_config.a_true,
#         "m_true": default_config.m_true,
#         "noise_var": default_config.noise_var,
#         "n_sampling_points": default_config.n_sampling_points,
#         "n_times": default_config.n_times,
#         "a_initialization": default_config.a_initialization,
#         "m_grid": default_config.m_grid,
#         "a_optimization": default_config.a_optimization,
#         "a_lr": default_config.a_lr,
#     },
# )

# config = wandb.config

# wandb.run.name = config.run_name

# logging.info(
#     f"Load dataset {config.dataset_name} with "
#     f"a_true = {config.a_true} and m_true = {config.m_true}"
# )
# dataset_of_trajectories = None
# if config.dataset_name == "synthetic_circle_to_ellipse":
#     if config.m_true == 1 and config.a_true == 1:
#         dataset_of_trajectories = synthetic.geodesics_circle_to_ellipse(
#             n_geodesics=1, n_times=config.n_times, n_points=config.n_sampling_points
#         )
# if dataset_of_trajectories is None:
#     raise NotImplementedError()

# one_trajectory = dataset_of_trajectories[0]
# print(f"The shape of one_trajectory is: {one_trajectory.shape}")

# if config.a_initialization == "close_to_ground_truth":
#     a_init = config.a_true - 0.2
# elif config.a_initialization == "random":
#     a_init = 0.5
# else:
#     raise NotImplementedError()

# logging.info("Find best a and m corresponding to the trajectory.")
# best_a, best_m, best_r2, r2, r2_srv, iteration_histories = optimize_am.find_best_am(
#     one_trajectory, a_init=a_init, m_grid=config.m_grid, a_lr=config.a_lr
# )

# logging.info("--->>> Save results in wandb and local saved_figs directory.")

# logging.info("1. Save the config locally.")
# config_df = pd.DataFrame.from_dict(dict(config))
# config_df.to_json(f"saved_figs/optimize_am/{config.run_name}_config.json")

# logging.info("2. Save best values for a, m and r2.")
# best_amr2_df = pd.DataFrame(
#     columns=["best_a", "best_m", "best_r2"], data=[[best_a, best_m, best_r2]]
# )

# r2s_from_m_df = pd.DataFrame(
#     columns=[f"m = {m}" for m in list(config.m_grid)], data=[list(r2), list(r2_srv)]
# )

# best_amr2_df.to_json(f"saved_figs/optimize_am/{config.run_name}_best_amr2.json")
# wandb.log({"best_amr2": wandb.Table(dataframe=best_amr2_df)})
# r2s_from_m_df.to_json(f"saved_figs/optimize_am/{config.run_name}_r2s_from_m_df.json")
# wandb.log({"r2s_from_m": wandb.Table(dataframe=r2s_from_m_df)})

# logging.info("3. Save iteration histories during gradient descent.")

# for i_m, m in enumerate(config.m_grid):
#     a_steps = iteration_histories[i_m]["a"]
#     mse_train_steps = iteration_histories[i_m]["mse_train"]
#     mse_val_steps = iteration_histories[i_m]["mse_val"]

#     r2_train_steps = iteration_histories[i_m]["r2_train"]
#     r2_val_steps = iteration_histories[i_m]["r2_val"]

#     iteration_history_df = pd.DataFrame(
#         columns=["a", "mse_train", "mse_val", "r2_train", "r2_val"],
#         data=[
#             [float(a), float(mse_train), float(mse_val), float(r_train), float(r_val)]
#             for a, mse_train, mse_val, r_train, r_val in zip(
#                 a_steps, mse_train_steps, mse_val_steps, r2_train_steps, r2_val_steps
#             )
#         ],
#     )

#     table_key = f"iteration_history_m_{m}"
#     iteration_history_df.to_json(
#         f"saved_figs/optimize_am/{config.run_name}_iteration_history.json"
#     )
#     wandb.log({table_key: wandb.Table(dataframe=iteration_history_df)})

# fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# for i_plot, plot_name in enumerate(["a", "mse", "r2"]):
#     if plot_name == "a":
#         axs[i_plot].axhline(config.a_true, label=f"a_true = {config.a_true}", c="black") # noqa: E501
#     for i_m, m in enumerate(config.m_grid):
#         if plot_name == "a":
#             iteration_history = iteration_histories[i_m][plot_name]
#             iterations = np.arange(0, len(iteration_history))
#             axs[i_plot].plot(
#                 iterations, iteration_history, label=f"m = {m}", c=f"C{m-1}"
#             )

#         elif plot_name in ["mse", "r2"]:
#             hval = 0 if plot_name == "mse" else 1
#             iteration_history = iteration_histories[i_m][plot_name + "_train"]
#             iterations = np.arange(0, len(iteration_history))
#             axs[i_plot].plot(
#                 iterations,
#                 iteration_history,
#                 label=f"m = {m} (train)",
#                 c=f"C{m-1}",
#                 linestyle="-",
#             )
#             iteration_history = iteration_histories[i_m][plot_name + "_val"]
#             iterations = np.arange(0, len(iteration_history))
#             axs[i_plot].plot(
#                 iterations,
#                 iteration_history,
#                 label=f"m = {m} (val)",
#                 c=f"C{m-1}",
#                 linestyle="--",
#             )

#             axs[i_plot].axhline(hval, c="black")
#     axs[i_plot].set_xlabel("Iterations")
#     axs[i_plot].set_title(plot_name)
#     axs[i_plot].legend()

# fig.suptitle(f"Optimization a, m gives: a = {best_a:.3f}, m = {best_m}, r2 = {best_r2}") # noqa: E501
# fig.savefig(f"saved_figs/optimize_am/{config.run_name}_iteration_history.png")
# wandb.log({"optimization_fig": wandb.Image(fig)})

# logging.info("4. Save plots of predicted curves.")
# # TODO.

# wandb.finish()
