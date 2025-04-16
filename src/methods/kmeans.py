import numpy as np
from scipy.stats import mode


class KMeans:
    def __init__(self, n_clusters=10, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.cluster_labels = None

    def init_centers(self, data):
        random_idx = np.random.permutation(data.shape[0])[:self.n_clusters]
        return data[random_idx]

    def compute_distance(self, data, centers):
        distances = np.zeros((data.shape[0], centers.shape[0]))
        for k in range(centers.shape[0]):
            distances[:, k] = np.linalg.norm(data - centers[k], axis=1)
        return distances

    def find_closest_cluster(self, distances):
        return np.argmin(distances, axis=1)

    def compute_centers(self, data, cluster_assignments):
        centers = np.zeros((self.n_clusters, data.shape[1]))
        for k in range(self.n_clusters):
            points = data[cluster_assignments == k]
            if points.shape[0] > 0:
                centers[k] = points.mean(axis=0)
            else:
                centers[k] = self.centers[k]  # re-use previous center if cluster is empty
        return centers

    def fit(self, data, labels=None):
        self.centers = self.init_centers(data)

        for i in range(self.max_iter):
            old_centers = self.centers.copy()
            distances = self.compute_distance(data, self.centers)
            cluster_assignments = self.find_closest_cluster(distances)
            self.centers = self.compute_centers(data, cluster_assignments)
            if np.allclose(old_centers, self.centers):
                break

        if labels is not None:
            self.cluster_labels = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                if np.any(cluster_assignments == k):
                    self.cluster_labels[k] = mode(labels[cluster_assignments == k], keepdims=True)[0]
            return self.cluster_labels[cluster_assignments]
        else:
            return cluster_assignments

    def predict(self, data):
        distances = self.compute_distance(data, self.centers)
        cluster_assignments = self.find_closest_cluster(distances)
        if self.cluster_labels is not None:
            return self.cluster_labels[cluster_assignments]
        return cluster_assignments
