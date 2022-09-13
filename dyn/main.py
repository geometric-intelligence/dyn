"""Main script."""

import logging
import os
import tempfile

import default_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

import dyn.dyn.datasets.synthetic as synthetic
import dyn.dyn.features.optimize_am as optimize_am

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

logging.info(f"Starting run {default_config.run_name}")
wandb.init(
    project="metric_learning",
    dir=tempfile.gettempdir(),
    config={
        "run_name": default_config.run_name,
        "dataset_name": default_config.dataset_name,
        "a_true": default_config.a_true,
        "m_true": default_config.m_true,
        "noise_var": default_config.noise_var,
        "n_sampling_points": default_config.n_sampling_points,
        "n_times": default_config.n_times,
        "a_initialization": default_config.a_initialization,
        "m_grid": default_config.m_grid,
        "a_optimization": default_config.a_optimization,
        "a_lr": default_config.a_lr,
    },
)

config = wandb.config

wandb.run.name = config.run_name

logging.info(
    f"Load dataset {config.dataset_name} with "
    f"a_true = {config.a_true} and m_true = {config.m_true}"
)
dataset_of_trajectories = None
if config.dataset_name == "synthetic_circle_to_ellipse":
    if config.m_true == 1 and config.a_true == 1:
        dataset_of_trajectories = synthetic.geodesics_circle_to_ellipse(
            n_geodesics=1, n_times=config.n_times, n_points=config.n_sampling_points
        )
if dataset_of_trajectories is None:
    raise NotImplementedError()

one_trajectory = dataset_of_trajectories[0]
print(f"The shape of one_trajectory is: {one_trajectory.shape}")

if config.a_initialization == "close_to_ground_truth":
    init_a = config.a_true - 0.2
elif config.a_initialization == "random":
    init_a = 0.5
else:
    raise NotImplementedError()

logging.info("Find best a and m corresponding to the trajectory.")
best_a, best_m, best_r2, r2, r2_srv, iteration_histories = optimize_am.find_best_am(
    one_trajectory, init_a=init_a, m_grid=config.m_grid, a_lr=config.a_lr
)

logging.info("--->>> Save results in wandb and local saved_figs directory.")

logging.info("1. Save the config locally.")
config_df = pd.DataFrame.from_dict(dict(config))
config_df.to_json(f"saved_figs/optimize_am/{config.run_name}_config.json")

logging.info("2. Save best values for a, m and r2.")
best_amr2_df = pd.DataFrame(
    columns=["best_a", "best_m", "best_r2"], data=[[best_a, best_m, best_r2]]
)

r2s_from_m_df = pd.DataFrame(
    columns=[f"m = {m}" for m in list(config.m_grid)], data=[list(r2), list(r2_srv)]
)

best_amr2_df.to_json(f"saved_figs/optimize_am/{config.run_name}_best_amr2.json")
wandb.log({"best_amr2": wandb.Table(dataframe=best_amr2_df)})
r2s_from_m_df.to_json(f"saved_figs/optimize_am/{config.run_name}_r2s_from_m_df.json")
wandb.log({"r2s_from_m": wandb.Table(dataframe=r2s_from_m_df)})

logging.info("3. Save iteration histories during gradient descent.")

for i_m, m in enumerate(config.m_grid):
    a_steps = iteration_histories[i_m]["a"]
    mse_train_steps = iteration_histories[i_m]["mse_train"]
    mse_val_steps = iteration_histories[i_m]["mse_val"]

    r2_train_steps = iteration_histories[i_m]["r2_train"]
    r2_val_steps = iteration_histories[i_m]["r2_val"]

    iteration_history_df = pd.DataFrame(
        columns=["a", "mse_train", "mse_val", "r2_train", "r2_val"],
        data=[
            [float(a), float(mse_train), float(mse_val), float(r_train), float(r_val)]
            for a, mse_train, mse_val, r_train, r_val in zip(
                a_steps, mse_train_steps, mse_val_steps, r2_train_steps, r2_val_steps
            )
        ],
    )

    table_key = f"iteration_history_m_{m}"
    iteration_history_df.to_json(
        f"saved_figs/optimize_am/{config.run_name}_iteration_history.json"
    )
    wandb.log({table_key: wandb.Table(dataframe=iteration_history_df)})

fig, axs = plt.subplots(1, 3, figsize=(20, 5))

for i_plot, plot_name in enumerate(["a", "mse", "r2"]):
    if plot_name == "a":
        axs[i_plot].axhline(config.a_true, label=f"a_true = {config.a_true}", c="black")
    for i_m, m in enumerate(config.m_grid):
        if plot_name == "a":
            iteration_history = iteration_histories[i_m][plot_name]
            iterations = np.arange(0, len(iteration_history))
            axs[i_plot].plot(
                iterations, iteration_history, label=f"m = {m}", c=f"C{m-1}"
            )

        elif plot_name in ["mse", "r2"]:
            hval = 0 if plot_name == "mse" else 1
            iteration_history = iteration_histories[i_m][plot_name + "_train"]
            iterations = np.arange(0, len(iteration_history))
            axs[i_plot].plot(
                iterations,
                iteration_history,
                label=f"m = {m} (train)",
                c=f"C{m-1}",
                linestyle="-",
            )
            iteration_history = iteration_histories[i_m][plot_name + "_val"]
            iterations = np.arange(0, len(iteration_history))
            axs[i_plot].plot(
                iterations,
                iteration_history,
                label=f"m = {m} (val)",
                c=f"C{m-1}",
                linestyle="--",
            )

            axs[i_plot].axhline(hval, c="black")
    axs[i_plot].set_xlabel("Iterations")
    axs[i_plot].set_title(plot_name)
    axs[i_plot].legend()

fig.suptitle(f"Optimization a, m gives: a = {best_a:.3f}, m = {best_m}, r2 = {best_r2}")
fig.savefig(f"saved_figs/optimize_am/{config.run_name}_iteration_history.png")
wandb.log({"optimization_fig": wandb.Image(fig)})

logging.info("4. Save plots of predicted curves.")
# TODO.

wandb.finish()
