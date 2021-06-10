CORESET_SIZE = 100
RESULTS_FILENAME = "Forest"
NUM_CLUSTERS = 7
NUM_LOOPS = 30

import random
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_kddcup99, fetch_covtype
import pandas as pd
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.metrics import pairwise_distances
from experiment_class import Experiment
from collections import Counter

start =time.time()
data, labels = fetch_covtype(return_X_y = True)
print(Counter(labels))
experiment = Experiment(RESULTS_FILENAME, data, NUM_CLUSTERS, CORESET_SIZE, NUM_LOOPS)
experiment.run_experiment()
print(time.time() - start, "seconds to run exp")