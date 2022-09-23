"""Main script."""

import itertools
import logging
import os
import tempfile

import default_config
import geomstats.backend as gs
import numpy as np
import pandas as pd
import wandb

import dyn.dyn.datasets.experimental as experimental
import dyn.dyn.datasets.synthetic as synthetic
import dyn.dyn.features.optimize_am as optimize_am
import dyn.dyn.viz as viz

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

np.random.seed(2022)
gs.random.seed(2022)

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
        noise_std,
        n_times,
        a_init_diff,
    ) in itertools.product(
        default_config.dataset_name,
        default_config.a_true,
        default_config.m_true,
        default_config.n_sampling_points,
        default_config.noise_std,
        default_config.n_times,
        default_config.a_init_diff,
    ):
        logging.info(
            f"Running tests for {dataset_name} with a_true={a_true}, m_true={m_true}:\n"
            f"- n_sampling_points={n_sampling_points}\n"
            f"- noise_std={noise_std}\n"
            f"- n_times={n_times}\n"
        )

        if dataset_name == "cells":
            cells = [default_config.start_cell, default_config.end_cell]
            _, cell_shapes, labels_a, labels_b = experimental.preprocess(
                cells=cells,
                labels_a=default_config.lines,
                labels_b=default_config.treatments,
                n_cells=2,
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
            noise_std=noise_std,
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
    noise_std,
    a_init,
    start_cell,
    end_cell,
):
    """Run wandb script for the following parameters."""
    run_name = (
        f"{dataset_name}_at{a_true}_ai{a_init}_mt{m_true}_"
        + f"nt{n_times}_nsp{n_sampling_points}_nv{noise_std}_{default_config.now}"
    )

    wandb.init(
        project="metric_learning",
        dir=tempfile.gettempdir(),
        config={
            "run_name": run_name,
            "dataset_name": dataset_name,
            "a_true": a_true,
            "m_true": m_true,
            "noise_std": noise_std,
            "n_sampling_points": n_sampling_points,
            "n_times": n_times,
            "percent_train": default_config.percent_train,
            "percent_val": default_config.percent_val,
            "a_init": a_init,
            "m_grid": default_config.m_grid,
            "a_optimization": default_config.a_optimization,
            "a_lr": default_config.a_lr,
            "max_iter": default_config.max_iter,
            "tol": default_config.tol,
        },
    )

    config = wandb.config
    wandb.run.name = config.run_name

    logging.info(
        f"Load dataset {dataset_name} with " f"a_true = {a_true} and m_true = {m_true}"
    )
    b = 0.5
    (
        noiseless_curve_traj,
        curve_traj,
        noiseless_q_traj,
        q_traj,
    ) = synthetic.geodesic_between_curves(
        start_cell, end_cell, a_true, b, n_times, noise_std
    )
    trajectory = curve_traj
    print(f"The shape of the trajectory is: {trajectory.shape}")

    n_times = len(trajectory)
    times = np.arange(0, n_times, 1)

    print("n_times: " + str(n_times))

    n_train = int(n_times * config.percent_train)
    n_val = int(n_times * config.percent_val)

    times_train = times[:n_train]  # noqa: E203
    times_val = times[n_train : (n_train + n_val)]  # noqa: E203
    times_test = times[(n_train + n_val) :]  # noqa: E203

    print(times_train)
    print(times_val)
    print(times_test)

    logging.info("Find best a and m corresponding to the trajectory.")
    (
        best_a,
        best_m,
        best_r2_val,
        r2_test_at_best,
        baseline_r2_srv_val,
        baseline_r2_srv_test,
        iteration_histories_per_i_m,
    ) = optimize_am.find_best_am(
        trajectory=trajectory,
        times_train=times_train,
        times_val=times_val,
        times_test=times_test,
        m_grid=config.m_grid,
        a_init=config.a_init,
        a_lr=config.a_lr,
        max_iter=config.max_iter,
        tol=config.tol,
    )
    logging.info("--->>> Save results in wandb and local saved_figs directory.")

    logging.info("1. Save the config locally.")
    config_df = pd.DataFrame.from_dict(dict(config))
    config_df.to_json(f"saved_figs/optimize_am/{config.run_name}_config.json")

    logging.info("2. Save iteration histories during gradient descent.")

    for i_m, m in enumerate(config.m_grid):
        a_steps = iteration_histories_per_i_m[i_m]["a"]
        mse_train_steps = iteration_histories_per_i_m[i_m]["mse_train"]
        mse_val_steps = iteration_histories_per_i_m[i_m]["mse_val"]

        r2_train_steps = iteration_histories_per_i_m[i_m]["r2_train"]
        r2_val_steps = iteration_histories_per_i_m[i_m]["r2_val"]

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

    fig = viz.plot_summary_wandb(
        iteration_histories_per_i_m=iteration_histories_per_i_m,
        config=config,
        noiseless_curve_traj=noiseless_curve_traj,
        curve_traj=curve_traj,
        noiseless_q_traj=noiseless_q_traj,
        q_traj=q_traj,
        times_train=times_train,
        times_val=times_val,
        times_test=times_test,
        best_a=best_a,
        best_m=best_m,
        best_r2_val=best_r2_val,
        r2_test_at_best=r2_test_at_best,
        baseline_r2_srv_val=baseline_r2_srv_val,
        baseline_r2_srv_test=baseline_r2_srv_test,
    )

    fig.savefig(f"saved_figs/optimize_am/{config.run_name}_summary.png")
    fig.savefig(f"saved_figs/optimize_am/{config.run_name}_summary.svg")
    wandb.log({"summary_fig": wandb.Image(fig)})

    wandb.finish()


run_tests()
