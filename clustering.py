import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


RANDOM_STATE = 44


# generate sparse matrix to ba able to use it in KMeans
def get_sparse_matrix(ratings):
    return csr_matrix((ratings.rate, (ratings.user, ratings.item)))


# Perform KMeans clustering
def kmeans(_n_clusters=10, _max_iter=500, _random_state=RANDOM_STATE, _verbose=0):
    return KMeans(n_clusters=_n_clusters,
                  max_iter=_max_iter,
                  random_state=_random_state,
                  verbose=_verbose
                  )


# return clusters centroids
def get_centroids(kmeans):
    centroids = pd.DataFrame(kmeans.cluster_centers_)
    return centroids


#  Return clusters sizes
def cluster_sizes(clusters, csr_ratings):
    size = {}
    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]

    for c in cluster_labels:
        size[c] = len(csr_ratings[clusters == c])
        print("Size of Cluster", c, "= ", size[c])

    return size


# Compute the Silhouette
def compute_silhouette():
    return 0