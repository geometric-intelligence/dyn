"""Default configuration for a run."""
import logging
from datetime import datetime

import numpy as np

import dyn.dyn.datasets.experimental as experimental

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Dataset
dataset_name = (
    #     "synthetic_circle_to_ellipse"
    "synthetic_geodesic_between_curves"
)
a_true = 1.0
m_true = 1

n_sampling_points = 50
n_times = 25
noise_var = 0.0

# Learning
# a_initialization = "close_to_ground_truth"
a_optimization = "gradient"
a_lr = 0.1
init_a = 0.8

# loading cells
n_cells = 650
n_sampling_points = 200
noise_var = 0
n_times = 20
quotient = ["scaling", "rotation"]
index_array = np.array([0, 10])
(
    cells,
    cell_shapes,
    labels_a,
    labels_b,
) = experimental.load_unrandomized_treated_osteosarcoma_cells(
    index_array, n_sampling_points=n_sampling_points, quotient=quotient
)
start_cell = cell_shapes[0]
end_cell = cell_shapes[1]

# m's
m_grid = [1, 2, 3]

# Run name in wandb
now = str(datetime.now().replace(second=0, microsecond=0).strftime("%m%d-%H:%M:%S"))
run_name = f"{now}_{dataset_name}"
