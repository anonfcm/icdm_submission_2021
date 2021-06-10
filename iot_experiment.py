CORESET_SIZE = 100
RESULTS_FILENAME = "IoT"
NUM_CLUSTERS = 11
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

# Some constants
PARENT_IOT_PATH = "iot/"

# path that contains all csv of benign traffic. 
BENIGN_PATH = PARENT_IOT_PATH + "benign_traffic"
BENIGN_CSV_FILES = ["benign_traffic.csv"]

# path that contains all csv of mirai attacks.
MIRAI_PATH = PARENT_IOT_PATH + "mirai_botnet"
# names of all the csv containing data about mirai attacks.
MIRAI_CSV = ["ack.csv", "scan.csv", "syn.csv", "udp.csv", "udpplain.csv"]


# path that contains all csv of gafgyt attacks.
GAFGYT_PATH = PARENT_IOT_PATH + "gafgyt_botnet"
# names of all the csv containing data about gafgyt attacks.
GAFGYT_CSV = ["combo.csv", "junk.csv", "scan.csv", "tcp.csv", "udp.csv"]

def get_benign_df():
	""" Code that gets the benign traffic csv and puts it into a dataframe."""
	benign_csv_path = BENIGN_PATH + "/" + BENIGN_CSV_FILES[0]
	benign_df = pd.read_csv(benign_csv_path)
	
	# create the feature list.
	benign_features = [BENIGN_CSV_FILES[0]]*len(benign_df)
	return benign_df, benign_features

def get_mirai_df(debug = False):
	""" Code that gets all df from mirai folder.
	Returns the features in df form and labels in py list form."""
	# for every file, we create a df
	feature_list = []
	for i,mirai_file in enumerate(MIRAI_CSV):
		current_file = MIRAI_PATH + "/" + mirai_file
		current_df = pd.read_csv(current_file)
		# add it to our complete mirai_df
		if i == 0:
			# if first mirai_df, nothing to add.
			mirai_df = current_df
			feature_list = feature_list + (['m' + mirai_file]*len(current_df))
			del current_df # for memory reasons.

		else:
			mirai_df = pd.concat([mirai_df, current_df])
			feature_list = feature_list + (['m' + mirai_file]*len(current_df))
			del current_df # for memory reasons.
	return mirai_df, feature_list

def get_gafgyt_df(debug = False):
	""" Code that gets all df from gafgyt folder.
	Returns the features in df form and labels in py list form."""
	# for every file, we create a df
	feature_list = []
	for i,gafgyt_file in enumerate(GAFGYT_CSV):
		current_file = GAFGYT_PATH + "/" + gafgyt_file
		current_df = pd.read_csv(current_file)
		# add it to our complete gafgyt df
		if i == 0:
			# if first gafgyt_df, nothing to add.
			gafgyt_df = current_df
			feature_list = feature_list + (['g' + gafgyt_file]*len(current_df))
			del current_df # for memory reasons.

		else:
			gafgyt_df = pd.concat([gafgyt_df, current_df])
			feature_list = feature_list + (['g' + gafgyt_file]*len(current_df))
			del current_df # for memory reasons.
	return gafgyt_df, feature_list	

def create_whole_dataset():
	# get required data.
	benign_df, benign_labels = get_benign_df()
	gafgyt_df, gafgyt_labels = get_gafgyt_df()
	mirai_df, mirai_labels = get_mirai_df()

	labels_list = []
	# debugging for writting purposes:
	print("benign labels:", set(benign_labels))
	print("gafgyt labels:", set(gafgyt_labels))
	print("mirai labels", set(mirai_labels))
	labels_list = benign_labels + gafgyt_labels + mirai_labels
	labels = np.array(labels_list)
	del benign_labels
	del gafgyt_labels
	del mirai_labels
	del labels_list

	features_df = pd.concat([benign_df, gafgyt_df, mirai_df])
	features = np.array(features_df)
	del benign_df
	del gafgyt_df
	del mirai_df
	print("number of data points:", len(features))
	print("number of labels", len(set(labels)))
	return features, labels


start =time.time()
data, labels = create_whole_dataset()
print(Counter(labels))
experiment = Experiment(RESULTS_FILENAME, data, NUM_CLUSTERS, CORESET_SIZE, NUM_LOOPS)
experiment.run_experiment()
