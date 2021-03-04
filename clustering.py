import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from statistics import mean

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
def cluster_sizes(csr_ratings, clusters, verbose=True):
    size = {}
    cluster_labels = np.unique(clusters)
    n_clusters = cluster_labels.shape[0]

    for c in cluster_labels:
        size[c] = csr_ratings[clusters == c].shape[0]
        if verbose:
            print("Size of Cluster", c, "= ", size[c])

    return size


# Compute the Silhouette
def get_silhouette(ratings, clusters):
    return metrics.silhouette_samples(ratings, clusters)


# create user vector for clustering purpose (a vector that contains all ratings for specific user)
def cluster_user_vector(user_profile, total_items):
    user_vector = []
    rated_items = {}  # items rated by user (Key: items_id, value: rate)

    for elm in user_profile:
        item_id = elm[0]
        rate = elm[1]
        rated_items[item_id] = rate

    for item_id in range(total_items):
        if item_id in rated_items.keys():
            user_vector.append(rated_items[item_id])
        else:
            user_vector.append(0)

    return user_vector


# Generate Top N recommedation list for each user based on neighbor cluster
def _top_n(clusters, centroids, user_profiles, total_items):
    for uid in range(len(clusters)):
        user_cluster = clusters[uid]

        # find all other users who are in the same cluster as target user
        neigh_users = np.where(clusters == user_cluster)

        user_vector = cluster_user_vector(user_profiles, uid, clusters.shape[0])
        neigh_vectors = []
        for neigh_id in neigh_users:
            neigh_vectors.append(cluster_user_vector(user_profiles, neigh_id, clusters.shape[0]))


# Get user vector and clusters centroid and return the most nearest cluster to the target user
def match_cluster(centroids, user_vec):
    sims = []
    for i in range(centroids.shape[0]):
        sim = cosine_similarity(np.array(centroids.iloc[i]).reshape(1, -1), np.array(user_vec).reshape(1, -1))
        sims.append(sim[0][0])

    return sims.index(max(sims))


def get_user_profile(data):
    unique_users = data.user.unique()
    unique_items = data.item.unique()

    user_profile = {}

    for id in unique_users:
        user = data[data.user == id]
        rated_items = []
        for index, row in user.iterrows():
            item_id = row[1]
            rate = row[2]

            # Create a tuple of new numeric item_id and its related rate for the current user id and
            # append this tuple to a list of rated items for the current user
            rated_items.append((int(item_id), int(rate)))

        user_profile[id] = rated_items

    return user_profile


# Generate Top N recommendations for users in testset based on neighbor cluster
def get_top_n(clusters, centroids, trainset, testset, N=10):
    top_n = {}
    train_user_profiles = get_user_profile(trainset)
    test_user_profiles = get_user_profile(testset)

    for uid in test_user_profiles:
        test_user_vec = cluster_user_vector(test_user_profiles[uid], centroids.shape[1])

        # Find the nearest cluster to the target test user
        neigh_cluster = match_cluster(centroids, test_user_vec)
        neigh_users = np.where(clusters == neigh_cluster)  # neighbor user ids
        neigh_users = neigh_users[0]
        rated_mat = np.empty((len(neigh_users), centroids.shape[1]))
        rated_mat[:] = np.NAN

        i = 0
        for neigh_id in neigh_users:
            rated_items = train_user_profiles[neigh_id]
            for item in rated_items:
                item_id = item[0]
                rate = item[1]
                rated_mat[i][item_id] = rate
            i += 1

        df = pd.DataFrame(data=rated_mat)
        temp = df.mean(axis=0).fillna(0)
        top_n[uid] = temp.argsort()[::-1][:N]

    return top_n


# Calculate Precision and Recall for Top-N recommendation task based on clustering at k
def precision_recall(top_n, relevant_items):
    precisions = []
    recalls = []
    for uid in top_n:
        recs = top_n[uid]  # list of recommendation for target test user
        rel = [x[0] for x in relevant_items[uid]]  # lsit of relevant items for target test user
        true_positive = len(set.intersection(set(recs), set(rel)))
        _precision = true_positive / len(recs) if len(recs) > 0 else 0
        _recall = true_positive / len(rel) if len(rel) > 0 else 0
        precisions.append(_precision)
        recalls.append(_recall)

    mean_pr = mean(precisions)
    mean_re = mean(recalls)
    f_measure = (2 * mean_pr * mean_re) / (mean_pr + mean_re)

    return mean_pr, mean_re, f_measure
