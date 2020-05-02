# Base libraries
import math
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
from scipy.special import erf
import pickle
import itertools

# Personal libraries
import sixtrackwrap as sx

import time

savepath = "/afs/cern.ch/work/c/camontan/public/sixtrack_da/"

maximum_time = 1.5 * 60.0 * 60.0 # one and a half hour in seconds

n_iterations = 50
sample = 100
step = 100

samples_in_batch = []
raw_data1 = []
raw_data2 = []
data1 = []
data2 = []

begin_time = time.time()

while (time.time() - begin_time < maximum_time):
    print("batch_size: ", sample)
    samples_in_batch.append(sample)
    
    # Simple tracking
    start = time.time()
    particles = sx.track_particles(np.zeros(sample), np.zeros(sample), np.zeros(sample), np.zeros(sample), n_iterations)
    tracking_time = time.time() - start
    print("simple:", tracking_time)
    raw_data1.append(tracking_time)
    data1.append(tracking_time / (sample * n_iterations))
    
    # Full tracking
    start = time.time()
    stuff = sx.full_track_particles(np.ones(sample), np.ones(sample), np.ones(sample), np.ones(sample), n_iterations)
    tracking_time = time.time() - start
    print("full: ", tracking_time)
    raw_data2.append(tracking_time)
    data2.append(tracking_time / (sample * n_iterations))
    
    # Increase step
    sample += step
    
with open(savepath + "tesla_benchmark.pkl", 'wb') as f:
    pickle.dump((samples_in_batch, raw_data1, raw_data2, data1, data2), f, protocol=4)