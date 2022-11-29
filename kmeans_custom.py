import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df = pd.read_csv('./diabetes.csv')

y = df['Outcome']
X = df[df.columns[:df.shape[1] - 1]]

K = 2
MAX_ITERS = 100

n_samples, n_features = X.shape
centroids = np.zeros((K, n_features))


def get_centroids(clusters):
    new_centroids = np.zeros((K, n_features))
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            new_centroids[cluster_idx] = centroids[cluster_idx]
            continue

        for feature_idx in range(n_features):
            rows = X.iloc[cluster]
            feature_mean = np.mean(rows[X.columns[feature_idx]])
            new_centroids[cluster_idx][feature_idx] = feature_mean

    return new_centroids


def create_clusters(centroids):
    clusters = [[] for _ in range(K)]
    for idx, sample in X.iterrows():
        centroid_idx = closest_centroid(sample, centroids)
        clusters[centroid_idx].append(idx)

    return clusters


def closest_centroid(sample, centroids):
    distances = np.zeros((K, n_features))
    for feat_idx, feature in enumerate(sample):
        # distance for centroid of first cluster
        distances[0][feat_idx] = centroids[0][feat_idx] - feature

        # distance for centroid of second cluster
        distances[1][feat_idx] = centroids[1][feat_idx] - feature

    distance1 = abs(np.mean(distances[0]))
    distance2 = abs(np.mean(distances[1]))

    return 0 if distance1 <= distance2 else 1


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def is_converged(old, curr):
    distances = [euclidean_distance(old[i], curr[i]) for i in range(K)]
    return sum(distances) == 0


# initialize centroids
for col_idx, column in enumerate(X.columns):
    centroids[0][col_idx] = random.uniform(X[column].min(), X[column].max())
    centroids[1][col_idx] = random.uniform(X[column].min(), X[column].max())

for _ in range(MAX_ITERS):
    # update clusters
    clusters = create_clusters(centroids)

    # update cluster centroids
    old_centroids = centroids
    centroids = get_centroids(clusters)

    # check if clusters have changed
    if is_converged(old_centroids, centroids):
        break

labels = np.empty(n_samples)
for cluster_idx, cluster in enumerate(clusters):
    for sample_idx in cluster:
        labels[sample_idx] = cluster_idx


TP = 0
FP = 0
TN = 0
FN = 0

for sample_idx, sample in X.iterrows():
    row = df.iloc[sample_idx]
    if sample_idx in clusters[1]:
        if row['Outcome'] == 1:
            TP += 1
        else:
            FP += 1
    else:
        if row['Outcome'] == 0:
            TN += 1
        else:
            FN += 1
