import numpy as np

min_turns = 100
max_turns = 50000
n_turn_samples = 100

turn_sampling = np.linspace(min_turns, max_turns, n_turn_samples, dtype=np.int_)[::-1]

d_r = 1.0
starting_step = 10 # USE IT CAREFULLY AS IT REQUIRES PRIOR KNOWLEDGE ON DA

batch_size = 20000

# BASELINE COMPUTING
baseline_samples = 33
baseline_total_samples = baseline_samples ** 3

# RADIAL AVERAGE COMPUTING
n_subdivisions = 16
samples = 1025

# MONTE CARLO
mc_max_samples = 10 ** 4
mc_min_samples = 10 ** 1
mc_samples = np.linspace(mc_min_samples, mc_max_samples, 1000, dtype=np.int)

# STRATIFIED MONTE CARLO
mcs_max_samples = 10 ** 4
mcs_samples = np.linspace(0, mcs_max_samples, 11, dtype=np.int)[1:]
mcs_n_sectors = 2

# 2D SAMPLING

alpha_max_samples = 2 ** 9 + 1