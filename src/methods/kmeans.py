import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500, n_clusters=5, n_init=10):
        self.max_iters = max_iters
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        best_accuracy = 0
        best_centroids = None
        best_pred_labels = None

        for _ in range(self.n_init):
            
            centroids = training_data[np.random.choice(training_data.shape[0], self.n_clusters, replace=False)]

            for _ in range(self.max_iters):
                
                distances = np.linalg.norm(training_data[:, np.newaxis] - centroids, axis=2)
                cluster_assignments = np.argmin(distances, axis=1)

                
                new_centroids = np.array([training_data[cluster_assignments == k].mean(axis=0) if np.any(cluster_assignments == k) else centroids[k] for k in range(self.n_clusters)])
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            
            pred_labels = np.zeros_like(cluster_assignments)
            for k in range(self.n_clusters):
                if np.any(cluster_assignments == k):
                    pred_labels[cluster_assignments == k] = mode(training_labels[cluster_assignments == k])[0]

            
            accuracy = np.mean(pred_labels == training_labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_centroids = centroids
                best_pred_labels = pred_labels

        self.centroids = best_centroids

        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        return best_pred_labels
    


    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments


