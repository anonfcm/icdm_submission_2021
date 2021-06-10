# code modified from holtskinner, implementation of FCM

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster._kmeans import _init_centroids

def d2_seeding(data, num_clusters):
    return _init_centroids(data, num_clusters, "k-means++")

def _update_clusters(x, u, m, weights = None):
    um = u ** m
    if weights:
        v = um.dot(x) * weights[:, None] / np.atleast_2d(um.sum(axis=1)).T * weights[: , None]
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


def _fcm_criterion(x, v, m, metric):

    d = cdist(x, v, metric=metric).T

    # Sanitize Distances (Avoid Zeroes)
    d = np.fmax(d, np.finfo(x.dtype).eps)

    exp = -2. / (m - 1)
    d2 = d ** exp

    u = d2 / np.sum(d2, axis=0, keepdims=1)
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
        print(u[t].T)
        t += 1

    # modified: centroids, initial centroids, membership, initial membership, ???, iterations
    return v[t], v[0], u[t - 1], u[0], d, t

# Public Facing Functions
def hcm(x, c, max_iterations = 100):
    return _cmeans(x, c, max_iterations, _hcm_criterion)

def fcm(x, c, max_iterations  = 100, m = 2):
    # inputs: data, num_clusters, m, tol, max_iterations, metric, initial centroids
    # returns: centroids, initial centroids, membership, initial membership, ???, iterations
    return _cmeans(x, c, max_iterations, _fcm_criterion, m = 2)
