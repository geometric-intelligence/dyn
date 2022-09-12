"""Default configuration for a run."""

import logging
from datetime import datetime

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Dataset
dataset_name = "synthetic_circle_to_ellipse"  # can be replaced later on with "experimental"
a_true = 1.
m_true = 1

n_sampling_points = 50
n_times = 25
noise_var = 0.

# Learning
a_initialization = "close_to_ground_truth"
a_optimization = "gradient"
a_lr = 0.05

m_grid = [1, 2]

# Run name in wandb
now = str(datetime.now().replace(second=0, microsecond=0))
run_name = f"{dataset_name}_{now}"