CORESET_SIZE = 100
RESULTS_FILENAME = "KDD"
NUM_CLUSTERS = 3
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

def get_kdd():
    data, labels = fetch_kddcup99(return_X_y = True, shuffle = True)
    print("num features before encoding:", len(data[0]))
    data = pd.DataFrame(data)
    data = data.drop([1,2,3,6, 11,20,21], axis = 1)
    data = np.array(data)
    return data
    
start =time.time()
data = get_kdd()
experiment = Experiment(RESULTS_FILENAME, data, NUM_CLUSTERS, CORESET_SIZE, NUM_LOOPS)
experiment.run_experiment()
print("One step closer to your own coreset")
print(time.time() - start, "seconds to run exp")