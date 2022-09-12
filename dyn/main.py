

"""Main script."""
import os
import tempfile

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import default_config
import matplotlib.pyplot as plt

import dyn.dyn.datasets.experimental as experimental
import dyn.dyn.datasets.synthetic as synthetic

import dyn.dyn.features.optimize_am as optimize_am


import wandb


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

dataset_of_trajectories = None
if config.dataset_name == "synthetic_circle_to_ellipse":
    if config.m_true == 1 and config.a_true == 1:
        dataset_of_trajectories = synthetic.geodesics_circle_to_ellipse(
            n_geodesics=1, n_times=config.n_times, n_points=config.n_sampling_points

    )
if dataset_of_trajectories is None:
    raise NotImplementedError()

one_trajectory = dataset_of_trajectories[0]
print(f"The shape of the trajectory of curves is: {one_trajectory.shape}")

if config.a_initialization == "close_to_ground_truth":
    init_a = config.a_true #- 0.2
elif config.a_initialization == "random":
    init_a = 0.5
else:
    raise NotImplementedError()

best_a, best_m, best_r2, best_r2_from_m, as_steps, r2s_steps = optimize_am.find_best_am(
    one_trajectory, init_a = init_a, m_grid =config.m_grid, a_lr = config.a_lr)

best_amr2 = wandb.Table(
    columns=["best_a", "best_m", "best_r2"], 
    data=[[best_a, best_m, best_r2]])

r2_from_m_results = wandb.Table(
    columns=[f"m = {m}" for m in list(config.m_grid)], 
    data=[list(best_r2_from_m)])

wandb.log(
        {"best_amr2": best_amr2, 
        "r2_from_m_results": r2_from_m_results})

for i_m, m in enumerate(config.m_grid):
    print(f"Log Table on optimization on a for m = {m}")
    a_r2_steps = wandb.Table(
        columns=["a", "r2"],
        data=[ [float(a), float(r)] for a, r in zip(as_steps[i_m], r2s_steps[i_m]) ])
    table_key = f"a_r2_steps_m_{m}"
    wandb.log(
        {table_key: a_r2_steps})

wandb.finish()
