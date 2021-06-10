# code from holtskinner, implementation of FCM
# TODO: we are trying to modify this so that the rows are datapoints
# and the columns are dimensions.

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster._kmeans import _init_centroids
from collections import Counter
import random
from sklearn.metrics import pairwise_distances

def get_voronoi_weights(data, samples):
    """
    Given some samples, weigh each prototype according to how many points
    in the big data set get assigned to it.
    """
    D = pairwise_distances(data, samples)
    closest = np.argmin(D, axis = 1)
    c = Counter(closest)
    ls = []
    for i in range(len(samples)):
        if i in c:
            ls.append(c[i])
        else:
            ls.append(0)
    return np.array(ls)


def maximin_sample(data, num_indices):
    """
    Returns a maximin sample.
    """

    # Start can be the first point or a random one. Does not really matter.
    starting_index = random.randint(0, len(data) - 1)
    # starting_index = 0
    mList = []
    latest_sampled_index = starting_index
    distance_list = cdist(data, [data[starting_index]])
    distance_list = np.linalg.norm(data - data[starting_index], axis = 1)
    for t in range(num_indices):
        newest_distances = np.linalg.norm(data - data[latest_sampled_index], axis = 1)
        distance_list = np.minimum(newest_distances, distance_list)
        latest_sampled_index = np.argmax(distance_list) 
        mList.append(latest_sampled_index)
    return data[mList]

def random_sample(data, n):
    # Returns a random sample of size n from the dataset (with replacement)
    return data[random.sample(range(len(data)), n)]

def mmdrs(data, num_maximin, num_sample):
    maximin_data = maximin_sample(data, num_maximin)
    # TODO
    # first we need to group the data into buckets. 
    # maybe have a membership vector by doing np.argmin of a distance matrix..
    # then, for i in range(len(set(list(membership_vector))))
    # we slice out the relevant buckets and add them to a list.
    # then for each bucket in that list.. 
    # n_t = ceiling(n * len(bucket) / len(data))
    # draw n_t samples from that bucket, then slice them out and add them to our list.
    # transform it into np.array() then return it.
    dist_matrix = cdist(data, maximin_data)
    memberships = [np.argmin(row) for row in dist_matrix]
    buckets = []
    for i in range(num_maximin):
        curr_bucket_idx = [j for j, membership in enumerate(memberships) if membership == i]
        # print("debug", curr_bucket_idx )
        current_bucket = data[curr_bucket_idx]
        buckets.append(current_bucket)
    sampled_data = []
    # print("debug", memberships)
    # print("debug", buckets)
    for bucket in buckets:
        n_t = int(np.ceil(num_sample * len(bucket) / len(data)))
        bucket_sample = random_sample(bucket, n_t)
        for point in bucket_sample:
            sampled_data.append(point)

    return np.array(sampled_data)

def d2_seeding(data, num_clusters):
    return _init_centroids(data, num_clusters, "k-means++")

def _update_clusters(x, u, m, weights = None):
    um = u ** m
    
    if weights is not None:
        v = um.dot(x * weights[:, None])   / np.atleast_2d((um * (weights.T)).sum(axis=1)).T
    else:
        v = um.dot(x)  / np.atleast_2d(um.sum(axis=1)).T 
    return v

def _hcm_criterion(x, v, m, metric):

    d = cdist(x, v, metric=metric)

    y = np.argmin(d, axis=1)

    u = np.zeros((v.shape[0], x.shape[0]))

    for i in range(x.shape[0]):
        u[y[i]][i] = 1

    return u, d

def fcm_error(dataset, centroids, m = 2, metric = "euclidean"):
    """
    Given a dataset, centroids, m and metric, returns FCM error.
    """
    # first calculate distance matrix and membership matrix.
    u, d = _fcm_criterion(dataset, centroids, m, metric)

    # then refer to u and d to get the error.
    error = 0
    for i in range(len(u)):
        for j in range(len(u[1])):
            error += u[i][j] * d[i][j]
    return error

def _fcm_criterion(x, v, m, metric):
    # x is N times p
    # v is c times p
    # memberships is c times N 
    # distance matrix is c times N
    d = cdist(x, v, metric=metric).T

    # Sanitize Distances (Avoid Zeroes)
    d = np.fmax(d, np.finfo(x.dtype).eps)

    exp = -2. / (m - 1)
    d2 = d ** exp

    u = d2 / np.sum(d2, axis=0, keepdims=1)
    # print("data shape", x.shape)
    # print("dist matrix", d.shape)
    # print("memberships", u.shape)
    # print("centroids shape", v.shape)
    # print()
    return u, d

def _cmeans(x, c, max_iterations, criterion_function, weights = None, metric="euclidean", v0=None, e = 0.0001, m = 2):
    num_points, num_features = x.shape
    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
    # If the user doesn't provide their own starting points,
    # we can have ++ initialization here?
    if v0 is None:
        # Pick random values from dataset
        # v0 = x[np.random.choice(num_points, c, replace=False), :]
        v0 = d2_seeding(x, c)
    # List of all cluster centers (Bookkeeping)
    v = np.empty((max_iterations, c, num_features))
    v[0] = np.array(v0)

    # Membership Matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, c, num_points))
    # Number of Iterations
    t = 0

    while t < max_iterations - 1:
        # calculate the next membership
        u[t], d = criterion_function(x, v[t], m, metric)
        # update the centroids
        v[t + 1] = _update_clusters(x, u[t], m, weights)
        # debug print(v)
        # Stopping Criteria
        if np.linalg.norm(v[t + 1] - v[t]) < e:
            break
        # debug
        # print(u[t].T)
        t += 1

    # modified: centroids, initial centroids, membership, initial membership, ???, iterations
    return v[t], v[0], u[t - 1], u[0], d, t

# Public Facing Functions
def hcm(x, c, max_iterations = 100):
    return _cmeans(x, c, max_iterations, _hcm_criterion)

def fcm(x, c, weights = None, max_iterations  = 100, m = 2):
    # inputs: data, num_clusters, m, tol, max_iterations, metric, initial centroids
    # returns: centroids, initial centroids, membership, initial membership, ???, iterations
    return _cmeans(x, c, max_iterations, _fcm_criterion,  weights, m = 2)
